"""Generación de código C++: estrategias sin memoización, descendente,
ascendente y ascendente con optimización de espacio (uno y dos parámetros)."""
from typing import List, Optional, Tuple
from modelo import *


def _pp_lhs_simple(llamada: Llamada) -> str:
    """Render sencillo del lado izquierdo de una ecuación (p. ej. 'mochila(i, c)')
    para mensajes de diagnóstico."""
    def arg(a):
        match a:
            case Numero(valor=v): return str(v)
            case Variable(nombre=n): return n
            case _: return "?"
    return f"{llamada.nombre}({', '.join(arg(a) for a in llamada.argumentos)})"


class CodeGenerator:
    """Genera C++ descendente (top-down) con memoización.

    A diferencia de la versión anterior, NO usa variables globales ni emite
    un `main`. Según la opción ``modo``:

      - ``"funcion"``: una pareja de funciones sobrecargadas. La pública
        recibe los datos del problema, reserva la tabla y delega en la
        privada (que añade la tabla por referencia) la recursión.
      - ``"clase"``: una clase con los datos y la tabla como atributos y un
        ``operator()`` que lanza la recursión.

    Los datos del problema (arrays y escalares) se pasan como parámetros /
    atributos, nunca como globales.
    """

    def __init__(self, modo: str = "funcion"):
        if modo not in ("funcion", "clase"):
            raise ValueError(f"modo de generación desconocido: {modo!r}")
        self.codigo: List[str] = []
        self.indent_level = 0
        self.modo = modo
        self.nombre_func = ""
        self.nombres_params: List[str] = []   # parámetros formales de la recurrencia
        self.arrays_all: List[Declaracion] = []
        self.scalars_all: List[Declaracion] = []
        self.datos_ref: List[Declaracion] = []  # declaraciones referenciadas en los cuerpos

    def emitir(self, linea: str) -> None:
        """Añade una línea con la indentación actual."""
        self.codigo.append("    " * self.indent_level + linea)

    def _preparar(self, programa: ProgramaDP) -> None:
        """Calcula el estado común a todos los generadores: nombre de la
        función, parámetros formales, datos (arrays/escalares) y qué datos
        están realmente referenciados en los cuerpos."""
        self.nombre_func = programa.retorno.nombre
        self.nombres_params = self._inferir_params_formales(programa)
        self.arrays_all = [d for d in programa.declaraciones if isinstance(d.tipo, ArrayType)]
        self.scalars_all = [d for d in programa.declaraciones if isinstance(d.tipo, BasicType)]
        referenciados = self._recolectar_referenciados(programa)
        self.datos_ref = [d for d in programa.declaraciones if d.nombre in referenciados]

    def _emitir_includes(self) -> None:
        """Bibliotecas (sin <iostream>: ya no se genera I/O ni main)."""
        self.emitir("#include <vector>")
        self.emitir("#include <algorithm>")
        self.emitir("#include <climits>")
        self.emitir("using namespace std;")
        self.emitir("")

    def generar(self, programa: ProgramaDP) -> str:
        """Punto de entrada. Organiza la creación del fichero C++ completo."""
        self._preparar(programa)
        self._emitir_includes()

        if self.modo == "clase":
            self._generar_clase(programa)
        else:
            self._generar_funciones(programa)

        return "\n".join(self.codigo)

    # --- Inferencia de estructura -----------------------------------------

    def _inferir_params_formales(self, programa: ProgramaDP) -> List[str]:
        """Nombres de los parámetros formales: los de la ecuación con MÁS
        nombres de variable DISTINTOS en su lado izquierdo.

        Contar nombres distintos (no solo Variables) evita elegir un caso
        base como `secMatrices(i, i)` —que repite `i`— frente al caso general
        `secMatrices(i, j)`, que es el que fija los parámetros reales."""
        nombres: List[str] = []
        mejor_distintos = -1
        for eq in programa.ecuaciones:
            vars_en_izq = [a.nombre for a in eq.izq.argumentos if isinstance(a, Variable)]
            distintos = len(set(vars_en_izq))
            if distintos > mejor_distintos:
                mejor_distintos = distintos
                nombres = vars_en_izq
        return nombres

    def _recolectar_referenciados(self, programa: ProgramaDP) -> set:
        """Identificadores declarados que aparecen en los cuerpos o condiciones
        de las ecuaciones (es decir, los datos que la recursión necesita)."""
        declarados = {d.nombre for d in programa.declaraciones}
        usados: set = set()

        def visitar(nodo):
            match nodo:
                case Variable(nombre=n, indices=idxs):
                    if n in declarados:
                        usados.add(n)
                    for idx in idxs:
                        visitar(idx)
                case OperacionBinaria(izq=izq, der=der):
                    visitar(izq); visitar(der)
                case Llamada(argumentos=args):
                    for a in args:
                        visitar(a)
                case Reduccion(rango=rg, argumentos=args, filtro=ft):
                    if rg is not None:
                        visitar(rg.limite_inf); visitar(rg.limite_sup)
                    if ft is not None:
                        visitar(ft)
                    for a in args:
                        visitar(a)
                case _:
                    pass

        for eq in programa.ecuaciones:
            visitar(eq.der)
            if eq.condicion is not None:
                visitar(eq.condicion)
        return usados

    def _valor_inicial_resultado(self, programa: ProgramaDP) -> str:
        """Valor de arranque del acumulador de la celda: INT_MIN si maximiza,
        0 si suma, INT_MAX en otro caso (minimización o sin reducción)."""
        tipo = self._tipo_reduccion(programa)
        if tipo == "max":
            return "INT_MIN"
        if tipo == "sum":
            return "0"
        return "INT_MAX"  # por defecto y para min

    def _tipo_reduccion(self, programa: ProgramaDP) -> Optional[str]:
        """Devuelve 'min'/'max' según la primera Reduccion encontrada, o None."""
        encontrado: Optional[str] = None

        def visitar(nodo):
            nonlocal encontrado
            if encontrado is not None:
                return
            match nodo:
                case Reduccion(tipo=t):
                    encontrado = t.lower()
                case OperacionBinaria(izq=izq, der=der):
                    visitar(izq); visitar(der)
                case Llamada(argumentos=args):
                    for a in args:
                        visitar(a)
                case _:
                    pass

        for eq in programa.ecuaciones:
            visitar(eq.der)
        return encontrado

    # --- Generadores de esqueleto -----------------------------------------

    def _generar_funciones(self, programa: ProgramaDP) -> None:
        """Modo función: helper recursivo (con memo) + función pública."""
        # Helper recursivo (sobrecarga con la tabla por referencia).
        self.emitir("// Función recursiva con memoización.")
        self.emitir(f"int {self.nombre_func}({self._params_helper_str()}) {{")
        self.indent_level += 1
        self._generar_cuerpo_resolucion(programa)
        self.indent_level -= 1
        self.emitir("}")
        self.emitir("")

        # Función pública: reserva la tabla y lanza la recursión.
        self.emitir("// Punto de entrada: reserva la tabla e inicia la recursión.")
        self.emitir(f"int {self.nombre_func}({self._params_publicos_str()}) {{")
        self.indent_level += 1
        tipo_memo = self._obtener_tipo_vector(len(self.nombres_params))
        self.emitir(f"{tipo_memo} memo;")
        self._generar_init_memo(programa)
        datos = [d.nombre for d in self.datos_ref]
        args_iniciales = [self.visit_Expresion(a) for a in programa.retorno.argumentos]
        llamada = ", ".join(datos + args_iniciales + ["memo"])
        self.emitir(f"return {self.nombre_func}({llamada});")
        self.indent_level -= 1
        self.emitir("}")

    def _generar_clase(self, programa: ProgramaDP) -> None:
        """Modo clase: datos y tabla como atributos; operator() público."""
        self.emitir(f"class {self.nombre_func} {{")
        self.indent_level += 1

        # Atributos: todas las declaraciones (en orden) + la tabla.
        for d in programa.declaraciones:
            self.emitir(f"{d.tipo.to_cpp()} {d.nombre};")
        tipo_memo = self._obtener_tipo_vector(len(self.nombres_params))
        self.emitir(f"{tipo_memo} memo;")
        self.emitir("")

        # Método privado recursivo.
        params = ", ".join(f"int {p}" for p in self.nombres_params)
        self.emitir(f"int resolver({params}) {{")
        self.indent_level += 1
        self._generar_cuerpo_resolucion(programa)
        self.indent_level -= 1
        self.emitir("}")

        self.indent_level -= 1
        self.emitir("")
        self.emitir("public:")
        self.indent_level += 1

        # Constructor: copia los datos y reserva la tabla.
        ctor_params, init_list = self._firma_constructor(programa)
        self.emitir(f"{self.nombre_func}({ctor_params}){init_list} {{")
        self.indent_level += 1
        self._generar_init_memo(programa)
        self.indent_level -= 1
        self.emitir("}")
        self.emitir("")

        # operator(): lanza la recursión con los parámetros de la recurrencia.
        op_params = ", ".join(f"int {p}" for p in self.nombres_params)
        args_canonicos = ", ".join(self.visit_Expresion(a) for a in programa.retorno.argumentos)
        self.emitir(f"int operator()({op_params}) {{")
        self.indent_level += 1
        self.emitir(f"// Llamada canónica del problema: ({args_canonicos})")
        self.emitir(f"return resolver({', '.join(self.nombres_params)});")
        self.indent_level -= 1
        self.emitir("}")

        self.indent_level -= 1
        self.emitir("};")

    def _generar_cuerpo_resolucion(self, programa: ProgramaDP, memoizar: bool = True) -> None:
        """Cuerpo común (casos base, recursión, return) compartido por el helper
        de modo función y el método `resolver` de modo clase. Si ``memoizar`` es
        False se omiten la consulta y el guardado en tabla (variante sin memo)."""
        self.emitir("// Casos Base")
        for eq in programa.ecuaciones:
            if eq.es_caso_base:
                self.visit_EcuacionBase(eq)
        self.emitir("")

        if memoizar:
            self.emitir("// Memoización")
            self._generar_check_memo()
            self.emitir("")

        self.emitir("// Casos Recursivos")
        valor_inicial = self._valor_inicial_resultado(programa)
        self.emitir(f"int resultado = {valor_inicial};")
        for eq in programa.ecuaciones:
            if not eq.es_caso_base:
                self.visit_EcuacionRecursiva(eq)
        self.emitir("")

        if memoizar:
            self._generar_return_memo()
        else:
            self.emitir("return resultado;")

    def _params_helper_str(self, con_memo: bool = True) -> str:
        """Parámetros del helper recursivo: datos referenciados + formales
        (+ memo si ``con_memo``)."""
        partes: List[str] = []
        for d in self.datos_ref:
            if isinstance(d.tipo, ArrayType):
                partes.append(f"const {d.tipo.to_cpp()}& {d.nombre}")
            else:
                partes.append(f"{d.tipo.to_cpp()} {d.nombre}")
        for p in self.nombres_params:
            partes.append(f"int {p}")
        if con_memo:
            tipo_memo = self._obtener_tipo_vector(len(self.nombres_params))
            partes.append(f"{tipo_memo}& memo")
        return ", ".join(partes)

    def _params_publicos_str(self) -> str:
        """Parámetros de la función pública: arrays (const ref) + escalares."""
        partes: List[str] = []
        for d in self.arrays_all:
            partes.append(f"const {d.tipo.to_cpp()}& {d.nombre}")
        for d in self.scalars_all:
            partes.append(f"{d.tipo.to_cpp()} {d.nombre}")
        return ", ".join(partes)

    def _firma_constructor(self, programa: ProgramaDP) -> Tuple[str, str]:
        """Devuelve (parámetros, lista de inicialización) del constructor de
        la clase: toma todas las declaraciones y las copia a los atributos."""
        params = ", ".join(f"{d.tipo.to_cpp()} {d.nombre}" for d in programa.declaraciones)
        if programa.declaraciones:
            init = " : " + ", ".join(f"{d.nombre}({d.nombre})" for d in programa.declaraciones)
        else:
            init = ""
        return params, init

    def _traducir_llamada(self, nodo: Llamada) -> str:
        """Punto único de traducción de una llamada (con acceso al AST, para
        que las subclases puedan inspeccionar los argumentos)."""
        args_str = [self.visit_Expresion(a) for a in nodo.argumentos]
        return self._render_llamada(nodo.nombre, args_str)

    def _render_llamada(self, nombre: str, args_str: List[str]) -> str:
        """Traduce una llamada del DSL a C++, propagando datos/tabla en las
        llamadas recursivas según el modo."""
        if nombre == self.nombre_func:
            if self.modo == "clase":
                return f"resolver({', '.join(args_str)})"
            datos = [d.nombre for d in self.datos_ref]
            return f"{self.nombre_func}({', '.join(datos + args_str + ['memo'])})"
        # Llamada a otra función (no debería ocurrir con una sola recurrencia).
        return f"{nombre}({', '.join(args_str)})"

    # --- MÉTODOS VISITOR (Traducción de Nodos a Strings) ---

    def visit_Reduccion(self, nodo) -> str:
        """Traduce reducciones min/max/sum, tanto discretas (lista de términos)
        como iterativas (con rango y bucle for)."""
        tipo = nodo.tipo.lower()

        # REDUCCIÓN DISCRETA (lista explícita de términos)
        if not nodo.rango:
            args_str = [self.visit_Expresion(arg) for arg in nodo.argumentos]
            if tipo == "sum":
                return "(" + " + ".join(args_str) + ")"
            operacion_cpp = "min" if tipo == "min" else "max"
            if len(args_str) == 2:
                return f"{operacion_cpp}({args_str[0]}, {args_str[1]})"
            elementos = ", ".join(args_str)
            return f"{operacion_cpp}({{{elementos}}})"

        # REDUCCIÓN ITERATIVA (con rango i <= k < j)
        rango = nodo.rango
        iterador = rango.iterador.nombre
        lim_inf = self.visit_Expresion(rango.limite_inf)
        lim_sup = self.visit_Expresion(rango.limite_sup)
        inicio_iter = lim_inf if rango.incluye_inf else f"{lim_inf} + 1"
        simbolo_der = "<=" if rango.incluye_sup else "<"
        condicion_iter = f"{iterador} {simbolo_der} {lim_sup}"
        expr_interna = self.visit_Expresion(nodo.argumentos[0])

        # Valor inicial (neutro) y forma de acumular y devolver, según el tipo.
        # El "devolver" trata el caso de conjunto vacío (rango vacío o filtro que
        # no deja pasar ningún k): para max es 0, para min es +inf (un centinela
        # finito grande, para que sumarle un coste no desborde), para sum es 0.
        if tipo == "sum":
            valor_inicial = "0"
            acumular = f"res_local = res_local + {expr_interna};"
            devolver = "res_local"
        elif tipo == "min":
            valor_inicial = "1000000000"   # +inf seguro (mayor que cualquier valor real)
            acumular = f"res_local = min(res_local, {expr_interna});"
            devolver = "res_local"
        else:  # max
            valor_inicial = "INT_MIN"
            acumular = f"res_local = max(res_local, {expr_interna});"
            devolver = "(res_local == INT_MIN ? 0 : res_local)"

        # Filtro opcional sobre el iterador: solo se agregan los k que lo cumplen.
        if nodo.filtro is not None:
            acumular = f"if ({self.visit_Expresion(nodo.filtro)}) {{ {acumular} }}"

        codigo_lambda = (
            f"[&]() {{ "
            f"int res_local = {valor_inicial}; "
            f"for (int {iterador} = {inicio_iter}; {condicion_iter}; {iterador}++) {{ "
            f"{acumular} "
            f"}} "
            f"return {devolver}; "
            f"}}()"
        )
        return codigo_lambda

    def visit_EcuacionBase(self, eq):
        """
        Traduce casos base. Ej: mochila(0, c) = 0; -> if (i == 0) return 0;
        """
        condiciones = self._condiciones_implicitas(eq)

        if eq.condicion:
            condiciones.append(self.visit_Expresion(eq.condicion))

        der_str = self.visit_Expresion(eq.der)
        if condiciones:
            cond_str = " && ".join(condiciones)
            self.emitir(f"if ({cond_str}) return {der_str};")
        else:
            self.emitir(f"return {der_str};")

    def _condiciones_implicitas(self, eq) -> list:
        """Extrae las condiciones implícitas del lado izquierdo de una ecuación.
        Ej: f(0, c) → ['i == 0'];   f(i, i) → ['j == i']."""
        condiciones = []
        for i, arg in enumerate(eq.izq.argumentos):
            param_formal = self.nombres_params[i]
            match arg:
                case Numero(valor=v):
                    condiciones.append(f"{param_formal} == {v}")
                case Variable(nombre=n) if n != param_formal:
                    condiciones.append(f"{param_formal} == {n}")
        return condiciones

    def visit_EcuacionRecursiva(self, eq):
        """
        Traduce casos recursivos, extrayendo tanto las condiciones explícitas (if)
        como las implícitas en los argumentos (ej: camino(i, 1)).
        """
        condiciones = self._condiciones_implicitas(eq)
                    
        # 2. Extraer condición explícita (el 'if' del DSL)
        if eq.condicion:
            condiciones.append(self.visit_Expresion(eq.condicion))
            
        der_str = self.visit_Expresion(eq.der)
        
        # Generamos el bloque IF si hay condición, o asignación directa si no
        if condiciones:
            cond_str = " && ".join(condiciones)
            self.emitir(f"if ({cond_str}) {{")
            self.indent_level += 1
            self.emitir(f"resultado = {der_str};")
            self.indent_level -= 1
            self.emitir(f"}}")
        else:
            self.emitir(f"resultado = {der_str};")

    def visit_Expresion(self, nodo) -> str:
        """
        Convierte cualquier nodo del AST en su representación de texto C++.
        Núcleo compartido por todos los generadores.
        """
        if isinstance(nodo, (int, str)):
            return str(nodo)

        match nodo:
            case Numero(valor=v):
                return str(v)

            case Variable(nombre=n, indices=idxs) if idxs:
                # Indexación 1-based uniforme: el índice se emite tal cual, sin
                # restar 1. Tanto los arrays como la tabla de memoización usan
                # la posición 0 como centinela (reservada para casos base), de
                # modo que los índices lógicos van de 1..n. Así las expresiones
                # C++ son idénticas a las de la recurrencia (p. ej. d[i-1]*d[k]
                # *d[j] del producto de matrices) y nunca hay doble resta.
                partes = [f"[{self.visit_Expresion(idx)}]" for idx in idxs]
                return f"{n}{''.join(partes)}"

            case Variable(nombre=n):
                return n

            case Llamada():
                return self._traducir_llamada(nodo)

            case OperacionBinaria(izq=izq, operador=op, der=der):
                op_cpp = {"and": "&&", "or": "||", "=": "=="}.get(op, op)
                return f"({self.visit_Expresion(izq)} {op_cpp} {self.visit_Expresion(der)})"

            case Reduccion():
                return self.visit_Reduccion(nodo)

            case _:
                return f"/* Error: nodo desconocido {type(nodo).__name__} */"

    def _obtener_tipo_vector(self, num_dims: int) -> str:
        """Genera el string del tipo C++ anidado. Ej (2): vector<vector<int>>"""
        return "vector<" * num_dims + "int" + ">" * num_dims

    def _generar_acceso_memo_local(self) -> str:
        """Genera el acceso a la tabla local usando los parámetros formales."""
        indices = "".join(f"[{param}]" for param in self.nombres_params)
        return f"memo{indices}"

    def _generar_check_memo(self):
        acceso = self._generar_acceso_memo_local()
        self.emitir(f"if ({acceso} != -1) return {acceso};")

    def _generar_return_memo(self):
        acceso = self._generar_acceso_memo_local()
        self.emitir(f"return {acceso} = resultado;")

    def _obtener_tamanos_memo(self, programa) -> list:
        """Tamaño de cada dimensión de la tabla (delegado en la función libre
        `inferir_tamanos_tabla`, compartida con el verificador de índices)."""
        return inferir_tamanos_tabla(programa)
    
    def _generar_assign_tabla(self, programa, nombre: str = "memo", relleno: str = "-1") -> None:
        """Genera el `.assign(...)` anidado de la tabla (`memo` o `tabla`) con
        los tamaños inferidos y el valor de relleno indicado."""
        tamanos = self._obtener_tamanos_memo(programa)
        num_dims = len(tamanos)

        def construir_anidado(idx):
            if idx == num_dims - 1:
                return f"{tamanos[idx]} + 1, {relleno}"
            tipo_interno = self._obtener_tipo_vector(num_dims - idx - 1)
            init_interno = construir_anidado(idx + 1)
            return f"{tamanos[idx]} + 1, {tipo_interno}({init_interno})"

        self.emitir(f"{nombre}.assign({construir_anidado(0)});")

    def _generar_init_memo(self, programa):
        """Compatibilidad: reserva la tabla `memo` rellena a -1 (top-down)."""
        self._generar_assign_tabla(programa, nombre="memo", relleno="-1")


class BottomUpGenerator(CodeGenerator):
    """Genera C++ ASCENDENTE (bottom-up): rellena iterativamente una tabla en
    un orden que respeta las dependencias entre celdas, sin recursión.

    Reutiliza los visitantes de `CodeGenerator` (traducción de expresiones,
    reducciones, condiciones). La diferencia esencial es que una llamada
    recursiva `f(args)` ya no es una llamada de función, sino una LECTURA de
    la tabla `tabla[args]` (ver `_render_llamada`).

    Soporta dos órdenes de llenado:
      - monótono: bucles `for` anidados, un parámetro por eje, en sentido
        creciente o decreciente según el análisis de dependencias;
      - por intervalos: cuando hay una reducción sobre un rango `i<=k<j`
        (p. ej. producto de matrices), se itera por longitud `j-i` creciente.
    """

    # --- Traducción de la llamada recursiva como acceso a la tabla --------

    def _render_llamada(self, nombre: str, args_str: List[str]) -> str:
        if nombre == self.nombre_func:
            return "tabla" + "".join(f"[{a}]" for a in args_str)
        return f"{nombre}({', '.join(args_str)})"

    # --- Punto de entrada -------------------------------------------------

    def generar(self, programa: ProgramaDP) -> str:
        self._preparar(programa)
        self._emitir_includes()
        if self.modo == "clase":
            self._generar_clase_bu(programa)
        else:
            self._generar_funcion_bu(programa)
        return "\n".join(self.codigo)

    # --- Esqueletos -------------------------------------------------------

    def _generar_funcion_bu(self, programa: ProgramaDP) -> None:
        self.emitir("// Algoritmo ascendente (bottom-up) con tabla completa.")
        self.emitir(f"int {self.nombre_func}({self._params_publicos_str()}) {{")
        self.indent_level += 1
        tipo = self._obtener_tipo_vector(len(self.nombres_params))
        self.emitir(f"{tipo} tabla;")
        self._generar_assign_tabla(programa, nombre="tabla", relleno="0")
        self._generar_llenado(programa)
        self.emitir(f"return {self.visit_Expresion(programa.retorno)};")
        self.indent_level -= 1
        self.emitir("}")

    def _generar_clase_bu(self, programa: ProgramaDP) -> None:
        self.emitir(f"class {self.nombre_func} {{")
        self.indent_level += 1
        for d in programa.declaraciones:
            self.emitir(f"{d.tipo.to_cpp()} {d.nombre};")
        tipo = self._obtener_tipo_vector(len(self.nombres_params))
        self.emitir(f"{tipo} tabla;")
        self.emitir("")
        self.emitir("void llenar() {")
        self.indent_level += 1
        self._generar_llenado(programa)
        self.indent_level -= 1
        self.emitir("}")
        self.indent_level -= 1
        self.emitir("")
        self.emitir("public:")
        self.indent_level += 1
        ctor_params, init_list = self._firma_constructor(programa)
        self.emitir(f"{self.nombre_func}({ctor_params}){init_list} {{")
        self.indent_level += 1
        self._generar_assign_tabla(programa, nombre="tabla", relleno="0")
        self.emitir("llenar();")
        self.indent_level -= 1
        self.emitir("}")
        self.emitir("")
        op_params = ", ".join(f"int {p}" for p in self.nombres_params)
        acceso = "tabla" + "".join(f"[{p}]" for p in self.nombres_params)
        self.emitir(f"int operator()({op_params}) {{ return {acceso}; }}")
        self.indent_level -= 1
        self.emitir("};")

    # --- Llenado de la tabla ---------------------------------------------

    def _generar_llenado(self, programa: ProgramaDP) -> None:
        if self._tiene_reduccion_con_rango(programa) and len(self.nombres_params) == 2:
            self.emitir("// Llenado por longitud de intervalo creciente (DP de intervalos).")
            self._emitir_nest_intervalo(programa)
        else:
            self.emitir("// Llenado por orden monótono de los índices.")
            self._emitir_nest_monotono(programa)

    def _emitir_nest_monotono(self, programa: ProgramaDP) -> None:
        tamanos = self._obtener_tamanos_memo(programa)
        for p, param in enumerate(self.nombres_params):
            size = tamanos[p]
            if self._direccion_axis(programa, p) == "dec":
                self.emitir(f"for (int {param} = {size}; {param} >= 0; {param}--) {{")
            else:
                self.emitir(f"for (int {param} = 0; {param} <= {size}; {param}++) {{")
            self.indent_level += 1
        self._emitir_cuerpo_celda(programa)
        for _ in self.nombres_params:
            self.indent_level -= 1
            self.emitir("}")

    def _emitir_nest_intervalo(self, programa: ProgramaDP) -> None:
        tamanos = self._obtener_tamanos_memo(programa)
        size = tamanos[-1]
        p0, p1 = self.nombres_params
        self.emitir(f"for (int longitud = 0; longitud <= {size} - 1; longitud++) {{")
        self.indent_level += 1
        self.emitir(f"for (int {p0} = 1; {p0} + longitud <= {size}; {p0}++) {{")
        self.indent_level += 1
        self.emitir(f"int {p1} = {p0} + longitud;")
        self._emitir_cuerpo_celda(programa)
        self.indent_level -= 1
        self.emitir("}")
        self.indent_level -= 1
        self.emitir("}")

    def _acceso_celda(self) -> str:
        """Lvalue donde se escribe el valor de la celda actual. Las subclases
        (p. ej. optimización de espacio) lo redefinen (curr[...])."""
        return "tabla" + "".join(f"[{p}]" for p in self.nombres_params)

    def _antes_de_ecuacion(self, eq: Ecuacion) -> None:
        """Hook que las subclases pueden usar antes de traducir cada ecuación
        (la versión con dos filas lo usa para resolver prev/curr)."""
        pass

    def _emitir_cuerpo_celda(self, programa: ProgramaDP) -> None:
        """Cuerpo común a cada celda: casos base (con `continue`) y luego los
        casos recursivos. El destino de escritura lo decide `_acceso_celda`."""
        acceso = self._acceso_celda()

        # Casos base: fijan la celda y saltan al siguiente índice.
        for eq in programa.ecuaciones:
            if not eq.es_caso_base:
                continue
            self._antes_de_ecuacion(eq)
            conds = self._condiciones_implicitas(eq)
            if eq.condicion is not None:
                conds.append(self.visit_Expresion(eq.condicion))
            val = self.visit_Expresion(eq.der)
            if conds:
                self.emitir(f"if ({' && '.join(conds)}) {{ {acceso} = {val}; continue; }}")
            else:
                self.emitir(f"{acceso} = {val}; continue;")

        # Casos recursivos.
        self.emitir(f"int resultado = {self._valor_inicial_resultado(programa)};")
        for eq in programa.ecuaciones:
            if eq.es_caso_base:
                continue
            self._antes_de_ecuacion(eq)
            conds = self._condiciones_implicitas(eq)
            if eq.condicion is not None:
                conds.append(self.visit_Expresion(eq.condicion))
            val = self.visit_Expresion(eq.der)
            if conds:
                self.emitir(f"if ({' && '.join(conds)}) {{")
                self.indent_level += 1
                self.emitir(f"resultado = {val};")
                self.indent_level -= 1
                self.emitir("}")
            else:
                self.emitir(f"resultado = {val};")
        self.emitir(f"{acceso} = resultado;")

    # --- Análisis de dependencias ----------------------------------------

    def _tiene_reduccion_con_rango(self, programa: ProgramaDP) -> bool:
        encontrado = False

        def walk(n):
            nonlocal encontrado
            match n:
                case Reduccion(rango=rg, argumentos=args):
                    if rg is not None:
                        encontrado = True
                    for a in args:
                        walk(a)
                case OperacionBinaria(izq=i, der=d):
                    walk(i); walk(d)
                case Llamada(argumentos=args):
                    for a in args:
                        walk(a)
                case _:
                    pass

        for eq in programa.ecuaciones:
            walk(eq.der)
        return encontrado

    def _llamadas_dp(self, nodo) -> List[Llamada]:
        """Todas las llamadas recursivas (a la propia función) dentro de un nodo."""
        res: List[Llamada] = []

        def walk(n):
            match n:
                case Llamada(nombre=nm, argumentos=args):
                    if nm == self.nombre_func:
                        res.append(n)
                    for a in args:
                        walk(a)
                case OperacionBinaria(izq=i, der=d):
                    walk(i); walk(d)
                case Reduccion(rango=rg, argumentos=args):
                    if rg is not None:
                        walk(rg.limite_inf); walk(rg.limite_sup)
                    for a in args:
                        walk(a)
                case Variable(indices=idxs):
                    for ix in idxs:
                        walk(ix)
                case _:
                    pass

        walk(nodo)
        return res

    def _delta(self, arg, formal: str) -> str:
        """Clasifica la relación del argumento recursivo con el formal:
        'igual', 'menor' (formal - algo), 'mayor' (formal + cte), u 'otro'."""
        match arg:
            case Variable(nombre=n, indices=idxs) if not idxs and n == formal:
                return "igual"
            case OperacionBinaria(operador="-", izq=Variable(nombre=n, indices=idxs)) \
                    if not idxs and n == formal:
                return "menor"
            case OperacionBinaria(operador="+", izq=Variable(nombre=n, indices=idxs), der=Numero()) \
                    if not idxs and n == formal:
                return "mayor"
            case OperacionBinaria(operador="+", der=Variable(nombre=n, indices=idxs), izq=Numero()) \
                    if not idxs and n == formal:
                return "mayor"
            case _:
                return "otro"

    def _direccion_axis(self, programa: ProgramaDP, p: int) -> str:
        """Sentido de iteración del eje p: 'inc' si las dependencias van a
        índices menores (lo habitual), 'dec' si van a mayores."""
        formal = self.nombres_params[p]
        menor = mayor = False
        for eq in programa.ecuaciones:
            if eq.es_caso_base:
                continue
            for ll in self._llamadas_dp(eq.der):
                if p < len(ll.argumentos):
                    d = self._delta(ll.argumentos[p], formal)
                    if d == "menor":
                        menor = True
                    elif d == "mayor":
                        mayor = True
        if mayor and not menor:
            return "dec"
        return "inc"


class SpaceOptGenerator(BottomUpGenerator):
    """Bottom-up con OPTIMIZACIÓN DE ESPACIO. Cubre dos casos:

    - **2 parámetros, ventana 1**: la celda (i, ·) solo depende de la fila
      actual y la anterior. Mantiene dos vectores fila (`prev`, `curr`) y los
      intercambia: O(tamaño de una fila) en lugar de O(tabla completa).
    - **1 parámetro, ventana acotada w**: la celda f(n) solo depende de
      f(n-1), …, f(n-w) con w constante. Mantiene una ventana deslizante de
      w+1 valores en un buffer circular indexado módulo (w+1): O(1) espacio
      (independiente de N). Es el caso de Fibonacci, factorial, Tribonacci.

    Si la recurrencia no es reducible (DP de intervalos, ventana variable como
    el corte de varilla, >2 parámetros…), deja un comentario y delega en el
    bottom-up con tabla completa, que siempre es correcto.
    """

    def __init__(self, modo: str = "funcion"):
        super().__init__(modo)
        self._fila_literal_actual: Optional[int] = None
        self._mod_1d: Optional[int] = None  # tamaño del buffer circular (1 parámetro)

    # --- Punto de entrada -------------------------------------------------

    def generar(self, programa: ProgramaDP) -> str:
        self._preparar(programa)

        # Caso de 1 parámetro con ventana acotada: buffer circular O(1).
        w = self._ventana_1d(programa)
        if w is not None:
            self._emitir_includes()
            self._mod_1d = w + 1
            if self.modo == "clase":
                self._generar_clase_envoltorio(programa, self._emitir_cuerpo_1d)
            else:
                self._generar_funcion_1d(programa)
            self._mod_1d = None
            return "\n".join(self.codigo)

        # Caso de 2 parámetros con ventana 1: dos filas.
        if not self._es_reducible(programa):
            # Delegamos en un BottomUpGenerator nuevo: así su traducción de
            # llamadas (tabla[...]) no se mezcla con la de esta clase (prev/curr).
            cuerpo = BottomUpGenerator(self.modo).generar(programa)
            nota = ("// [optimización de espacio] La recurrencia no admite reducción;\n"
                    "// se mantiene la tabla completa (bottom-up).\n")
            return nota + cuerpo

        self._emitir_includes()
        if self.modo == "clase":
            self._generar_clase_envoltorio(programa, self._emitir_cuerpo_so)
        else:
            self._generar_funcion_so(programa)
        return "\n".join(self.codigo)

    # --- Análisis y generación del caso de 1 parámetro -------------------

    def _ventana_1d(self, programa: ProgramaDP) -> Optional[int]:
        """Si la recurrencia es de UN parámetro y toda lectura recursiva es
        f(n - c) con c constante ≥ 1 (ventana acotada), devuelve w = máx c.
        Si hay una lectura de ventana variable (p. ej. f(n-k) en una reducción
        sobre rango) o no decreciente, devuelve None."""
        if len(self.nombres_params) != 1:
            return None
        formal = self.nombres_params[0]
        w = 0
        for eq in programa.ecuaciones:
            if eq.es_caso_base:
                continue
            for ll in self._llamadas_dp(eq.der):
                c = self._desplazamiento_const(ll.argumentos[0], formal)
                if c is None or c < 1:
                    return None
                w = max(w, c)
        return w if w >= 1 else None

    def _desplazamiento_const(self, arg, formal: str) -> Optional[int]:
        """Para f(formal - c) con c literal devuelve c; en otro caso None."""
        match arg:
            case OperacionBinaria(operador="-", izq=Variable(nombre=n, indices=idxs),
                                  der=Numero(valor=v)) if not idxs and n == formal:
                return v
            case _:
                return None

    def _generar_funcion_1d(self, programa: ProgramaDP) -> None:
        self.emitir(f"// Optimización de espacio O(1): ventana deslizante de {self._mod_1d} valores.")
        self.emitir(f"int {self.nombre_func}({self._params_publicos_str()}) {{")
        self.indent_level += 1
        self._emitir_cuerpo_1d(programa)
        self.indent_level -= 1
        self.emitir("}")

    def _emitir_cuerpo_1d(self, programa: ProgramaDP, guardar: Optional[str] = None) -> None:
        """Cuerpo con buffer circular de tamaño w+1. Cada índice de tabla se
        toma módulo w+1, de modo que los w+1 valores más recientes coexisten."""
        m = self._mod_1d
        p0 = self.nombres_params[0]
        size0 = inferir_tamanos_tabla(programa)[0]

        self.emitir(f"vector<int> tabla({m}, 0);")
        self.emitir(f"for (int {p0} = 0; {p0} <= {size0}; {p0}++) {{")
        self.indent_level += 1
        self._emitir_cuerpo_celda(programa)  # usa _acceso_celda y _traducir_llamada en modo 1D
        self.indent_level -= 1
        self.emitir("}")

        idx0 = self.visit_Expresion(programa.retorno.argumentos[0])
        destino = f"return tabla[({idx0}) % {m}];" if guardar is None else f"{guardar} = tabla[({idx0}) % {m}];"
        self.emitir(destino)

    def _generar_clase_envoltorio(self, programa: ProgramaDP, emitir_cuerpo) -> None:
        """Envoltorio de clase común a ambas variantes de space-opt: atributos,
        método privado `calcular()` con el cuerpo dado, constructor y un
        `operator()` sin argumentos que devuelve el resultado ya calculado."""
        self.emitir(f"class {self.nombre_func} {{")
        self.indent_level += 1
        for d in programa.declaraciones:
            self.emitir(f"{d.tipo.to_cpp()} {d.nombre};")
        self.emitir("int resultado_final;")
        self.emitir("")
        self.emitir("void calcular() {")
        self.indent_level += 1
        emitir_cuerpo(programa, guardar="resultado_final")
        self.indent_level -= 1
        self.emitir("}")
        self.indent_level -= 1
        self.emitir("")
        self.emitir("public:")
        self.indent_level += 1
        ctor_params, init_list = self._firma_constructor(programa)
        self.emitir(f"{self.nombre_func}({ctor_params}){init_list} {{ calcular(); }}")
        self.emitir("int operator()() { return resultado_final; }")
        self.indent_level -= 1
        self.emitir("};")

    # --- Análisis de reducibilidad ---------------------------------------

    def _es_reducible(self, programa: ProgramaDP) -> bool:
        """Reducible ⇔ exactamente 2 parámetros, sin DP de intervalos, y toda
        lectura recursiva cae en la fila actual (Δ=0) o la anterior (Δ=1)."""
        if len(self.nombres_params) != 2:
            return False
        if self._tiene_reduccion_con_rango(programa):
            return False
        max_delta = 0
        for eq in programa.ecuaciones:
            if eq.es_caso_base:
                continue
            fila_lit = self._fila_literal(eq)
            for ll in self._llamadas_dp(eq.der):
                d = self._delta_fila(ll.argumentos[0], fila_lit)
                if d is None or d < 0:
                    return False
                max_delta = max(max_delta, d)
        return max_delta == 1

    def _fila_literal(self, eq: Ecuacion) -> Optional[int]:
        """Si el LHS fija el primer parámetro a un literal L, devuelve L; si
        usa la variable formal, devuelve None."""
        arg0 = eq.izq.argumentos[0]
        if isinstance(arg0, Numero):
            return arg0.valor
        return None

    def _delta_fila(self, arg, fila_literal: Optional[int]) -> Optional[int]:
        """Desplazamiento en filas del primer índice de una llamada recursiva
        respecto a la fila de la celda actual. 0 = misma fila, 1 = anterior."""
        formal0 = self.nombres_params[0]
        match arg:
            case Variable(nombre=n, indices=idxs) if not idxs and n == formal0:
                return 0
            case OperacionBinaria(operador="-", izq=Variable(nombre=n, indices=idxs),
                                  der=Numero(valor=v)) if not idxs and n == formal0:
                return v
            case Numero(valor=m):
                if fila_literal is not None:
                    return fila_literal - m
                return None
            case _:
                return None

    # --- Traducción de la llamada recursiva como prev/curr ---------------

    def _traducir_llamada(self, nodo: Llamada) -> str:
        if nodo.nombre == self.nombre_func:
            if self._mod_1d is not None:
                # Buffer circular: f(e) → tabla[(e) % (w+1)].
                idx0 = self.visit_Expresion(nodo.argumentos[0])
                return f"tabla[({idx0}) % {self._mod_1d}]"
            delta = self._delta_fila(nodo.argumentos[0], self._fila_literal_actual)
            buffer = "curr" if delta == 0 else "prev"
            if len(nodo.argumentos) == 1:
                return buffer
            idx1 = self.visit_Expresion(nodo.argumentos[1])
            return f"{buffer}[{idx1}]"
        return f"{nodo.nombre}({', '.join(self.visit_Expresion(a) for a in nodo.argumentos)})"

    # --- Esqueletos -------------------------------------------------------

    def _generar_funcion_so(self, programa: ProgramaDP) -> None:
        self.emitir("// Algoritmo ascendente con optimización de espacio (dos filas).")
        self.emitir(f"int {self.nombre_func}({self._params_publicos_str()}) {{")
        self.indent_level += 1
        self._emitir_cuerpo_so(programa)
        self.indent_level -= 1
        self.emitir("}")

    def _emitir_cuerpo_so(self, programa: ProgramaDP, guardar: Optional[str] = None) -> None:
        """Cuerpo del cálculo con dos filas. Si `guardar` se indica, almacena
        el resultado final en ese atributo en vez de hacer `return`."""
        tamanos = self._obtener_tamanos_memo(programa)
        p0, p1 = self.nombres_params
        size0, size1 = tamanos[0], tamanos[1]

        self.emitir(f"vector<int> prev({size1} + 1, 0), curr({size1} + 1, 0);")
        self.emitir(f"for (int {p0} = 0; {p0} <= {size0}; {p0}++) {{")
        self.indent_level += 1
        self.emitir(f"for (int {p1} = 0; {p1} <= {size1}; {p1}++) {{")
        self.indent_level += 1
        self._emitir_cuerpo_celda(programa)   # reutiliza el cuerpo de BottomUp vía hooks
        self.indent_level -= 1
        self.emitir("}")
        self.emitir("swap(prev, curr);")  # la fila recién calculada pasa a ser 'prev'
        self.indent_level -= 1
        self.emitir("}")

        idx1 = self.visit_Expresion(programa.retorno.argumentos[1])
        if guardar is not None:
            self.emitir(f"{guardar} = prev[{idx1}];")
        else:
            self.emitir(f"return prev[{idx1}];")

    # Hooks que adaptan el cuerpo común de BottomUpGenerator.
    def _acceso_celda(self) -> str:
        if self._mod_1d is not None:
            return f"tabla[({self.nombres_params[0]}) % {self._mod_1d}]"
        return f"curr[{self.nombres_params[1]}]"

    def _antes_de_ecuacion(self, eq: Ecuacion) -> None:
        # Necesario para que _traducir_llamada resuelva prev/curr según la fila.
        self._fila_literal_actual = self._fila_literal(eq)


class SinMemoGenerator(CodeGenerator):
    """Recursión directa SIN memoización: traducción literal de la recurrencia.

    Es el algoritmo de coste (normalmente) exponencial que sirve de punto de
    partida y de referencia para comparar con top-down y bottom-up en la
    evaluación experimental. Reutiliza todo el cuerpo del generador top-down,
    pero sin tabla: ni consulta ni guardado, y las llamadas recursivas no
    propagan ninguna estructura de memoria.
    """

    def _render_llamada(self, nombre: str, args_str: List[str]) -> str:
        if nombre == self.nombre_func:
            if self.modo == "clase":
                return f"resolver({', '.join(args_str)})"
            datos = [d.nombre for d in self.datos_ref]
            return f"{self.nombre_func}({', '.join(datos + args_str)})"
        return f"{nombre}({', '.join(args_str)})"

    def generar(self, programa: ProgramaDP) -> str:
        self._preparar(programa)
        self._emitir_includes()
        if self.modo == "clase":
            self._generar_clase_sm(programa)
        else:
            self._generar_funcion_sm(programa)
        return "\n".join(self.codigo)

    def _generar_funcion_sm(self, programa: ProgramaDP) -> None:
        self.emitir("// Recursión directa SIN memoización (referencia; coste exponencial).")
        datos = [d.nombre for d in self.datos_ref]
        args_canonicos = [self.visit_Expresion(a) for a in programa.retorno.argumentos]
        llamada = ", ".join(datos + args_canonicos)
        self.emitir(f"// Llamada inicial del problema: {self.nombre_func}({llamada})")
        self.emitir(f"int {self.nombre_func}({self._params_helper_str(con_memo=False)}) {{")
        self.indent_level += 1
        self._generar_cuerpo_resolucion(programa, memoizar=False)
        self.indent_level -= 1
        self.emitir("}")

    def _generar_clase_sm(self, programa: ProgramaDP) -> None:
        self.emitir(f"class {self.nombre_func} {{")
        self.indent_level += 1
        for d in programa.declaraciones:
            self.emitir(f"{d.tipo.to_cpp()} {d.nombre};")
        self.emitir("")
        params = ", ".join(f"int {p}" for p in self.nombres_params)
        self.emitir(f"int resolver({params}) {{")
        self.indent_level += 1
        self._generar_cuerpo_resolucion(programa, memoizar=False)
        self.indent_level -= 1
        self.emitir("}")
        self.indent_level -= 1
        self.emitir("")
        self.emitir("public:")
        self.indent_level += 1
        ctor_params, init_list = self._firma_constructor(programa)
        self.emitir(f"{self.nombre_func}({ctor_params}){init_list} {{}}")
        self.emitir("")
        op_params = ", ".join(f"int {p}" for p in self.nombres_params)
        args_canonicos = ", ".join(self.visit_Expresion(a) for a in programa.retorno.argumentos)
        self.emitir(f"int operator()({op_params}) {{")
        self.indent_level += 1
        self.emitir(f"// Llamada canónica del problema: ({args_canonicos})")
        self.emitir(f"return resolver({', '.join(self.nombres_params)});")
        self.indent_level -= 1
        self.emitir("}")
        self.indent_level -= 1
        self.emitir("};")


class ReconstruccionGenerator(BottomUpGenerator):
    """Bottom-up con tabla completa MÁS reconstrucción de la solución óptima.

    Además de la función/clase de valor (idéntica al bottom-up), emite una
    reconstrucción que sigue la DECISIÓN ÓPTIMA (el argmin/argmax de la
    recurrencia de Bellman) en cada celda, leída recomputándola sobre la tabla
    de valores ya llena. Un ÚNICO mecanismo —un descenso recursivo que sigue esa
    decisión— cubre las dos formas que puede tener el óptimo:

      - **camino** (cada término referencia UN subproblema): el descenso es
        lineal y la salida es la SECUENCIA DE ESTADOS (celdas) del camino
        óptimo, de la llamada inicial al caso base. Cubre mochila, LCS, edición,
        monedas, varilla y caminos en rejilla.
      - **árbol / intervalos** (un término dentro de una reducción con rango
        referencia DOS subproblemas, p. ej. el producto de matrices): el
        descenso se ramifica en los dos subintervalos y la salida es la
        PARENTIZACIÓN óptima como lista de cortes `{i, j, k}` —el intervalo
        [i, j] se parte en k— en preorden del árbol de decisiones.

    En ambos casos la decisión se identifica recomputando cada rama (cada
    término de un min/max, o cada valor del iterador de una reducción) y
    comparándola con `tabla[celda]`; nunca se emite un resultado incorrecto: si
    la recurrencia no encaja en ninguna de las dos formas, se rechaza con un
    mensaje claro.

    Es incompatible con la optimización de espacio: el descenso necesita la
    tabla completa, no solo las dos últimas filas.
    """

    # --- Punto de entrada -------------------------------------------------

    def generar(self, programa: ProgramaDP) -> str:
        self._preparar(programa)
        self._comprobar_elegible(programa)
        self._emitir_includes()
        if self.modo == "clase":
            self._generar_clase_recon(programa)
        else:
            self._generar_funcion_recon(programa)
        return "\n".join(self.codigo)

    def _emitir_includes(self) -> None:
        # La reconstrucción usa una lambda recursiva (std::function).
        self.emitir("#include <vector>")
        self.emitir("#include <algorithm>")
        self.emitir("#include <climits>")
        self.emitir("#include <functional>")
        self.emitir("using namespace std;")
        self.emitir("")

    # --- Elegibilidad: camino (un subproblema) o intervalos (dos) ---------

    def _comprobar_elegible(self, programa: ProgramaDP) -> None:
        """Clasifica la recurrencia en CAMINO (un subproblema por término →
        secuencia) o ÁRBOL de intervalos (dos subproblemas dentro de una
        reducción con rango → parentización), fijando `self._es_arbol`. Rechaza
        con un mensaje claro lo que no encaje en ninguna de las dos formas, de
        modo que nunca se genere una reconstrucción incorrecta."""
        if self._tipo_reduccion(programa) == "sum":
            raise ValueError(
                "[Reconstrucción] La reconstrucción solo está definida para "
                "recurrencias de optimización (min/max); una suma (sum) no "
                "selecciona una decisión que reconstruir.")
        max_llamadas = 0
        for eq in programa.ecuaciones:
            if eq.es_caso_base:
                continue
            for alt in self._alternativas(eq):
                n = len(self._llamadas_dp(alt))
                if n == 0:
                    raise ValueError(
                        f"[Reconstrucción] Una rama de "
                        f"'{_pp_lhs_simple(eq.izq)}' no referencia ningún "
                        f"subproblema: no hay nada que reconstruir.")
                max_llamadas = max(max_llamadas, n)

        if max_llamadas == 1:
            self._es_arbol = False
        elif max_llamadas == 2 and self._es_patron_intervalo(programa):
            self._es_arbol = True
        else:
            raise ValueError(
                f"[Reconstrucción] La recurrencia no es reconstruible: sus "
                f"términos referencian {max_llamadas} subproblemas con una forma "
                f"no soportada. Se admiten dos casos: UN subproblema por término "
                f"(camino: mochila, LCS, edición, monedas, varilla, caminos) o "
                f"DOS subproblemas dentro de una reducción con rango sobre dos "
                f"parámetros (intervalos: producto de matrices).")

    def _es_patron_intervalo(self, programa: ProgramaDP) -> bool:
        """Patrón de DP de intervalos (tipo producto de matrices): dos
        parámetros y una única ecuación recursiva cuyo cuerpo es una reducción
        con rango cuyo término referencia exactamente dos subproblemas. Esto
        garantiza un iterador de corte `k` y el orden de llenado por longitud
        creciente, que deja ambos subintervalos ya calculados; el descenso que
        sigue la decisión es entonces correcto siempre que lo sea el DP de
        valor."""
        if len(self.nombres_params) != 2:
            return False
        recursivas = [eq for eq in programa.ecuaciones if not eq.es_caso_base]
        if len(recursivas) != 1:
            return False
        red = self._encontrar_reduccion(recursivas[0].der)
        if red is None or red.rango is None:
            return False
        cuerpo = self._sustituir_nodo(recursivas[0].der, red, red.argumentos[0])
        return len(self._llamadas_dp(cuerpo)) == 2

    def _alternativas(self, eq: Ecuacion) -> List[Expresion]:
        """Las expresiones-alternativa de una ecuación: una por término del
        min/max (sustituida en su contexto), o la propia RHS si no hay
        reducción. Para una reducción con rango, la única alternativa es el
        cuerpo (parametrizado por el iterador)."""
        red = self._encontrar_reduccion(eq.der)
        if red is None:
            return [eq.der]
        if red.rango is None:
            return [self._sustituir_nodo(eq.der, red, term) for term in red.argumentos]
        return [self._sustituir_nodo(eq.der, red, red.argumentos[0])]

    def _encontrar_reduccion(self, nodo) -> Optional[Reduccion]:
        """Primera Reduccion (min/max) en el árbol, o None."""
        match nodo:
            case Reduccion():
                return nodo
            case OperacionBinaria(izq=izq, der=der):
                return self._encontrar_reduccion(izq) or self._encontrar_reduccion(der)
            case Llamada(argumentos=args):
                for a in args:
                    r = self._encontrar_reduccion(a)
                    if r is not None:
                        return r
                return None
            case Variable(indices=idxs):
                for ix in idxs:
                    r = self._encontrar_reduccion(ix)
                    if r is not None:
                        return r
                return None
            case _:
                return None

    def _sustituir_nodo(self, nodo, objetivo, reemplazo):
        """Copia de `nodo` con el subárbol `objetivo` (por identidad) sustituido
        por `reemplazo`. Conserva el contexto (p. ej. el `+1` de la edición)."""
        if nodo is objetivo:
            return reemplazo
        match nodo:
            case OperacionBinaria(izq=izq, operador=op, der=der):
                return OperacionBinaria(
                    izq=self._sustituir_nodo(izq, objetivo, reemplazo),
                    operador=op,
                    der=self._sustituir_nodo(der, objetivo, reemplazo))
            case Llamada(nombre=nm, argumentos=args):
                return Llamada(nombre=nm, argumentos=[
                    self._sustituir_nodo(a, objetivo, reemplazo) for a in args])
            case Reduccion(tipo=t, rango=rg, argumentos=args, filtro=ft):
                return Reduccion(tipo=t, rango=rg, filtro=ft, argumentos=[
                    self._sustituir_nodo(a, objetivo, reemplazo) for a in args])
            case Variable(nombre=n, indices=idxs, tipo=tp):
                return Variable(nombre=n, tipo=tp, indices=[
                    self._sustituir_nodo(ix, objetivo, reemplazo) for ix in idxs])
            case _:
                return nodo  # Numero u otra hoja

    # --- Esqueletos: valor + reconstrucción -------------------------------

    def _generar_funcion_recon(self, programa: ProgramaDP) -> None:
        # 1) Función de valor (bottom-up con tabla completa).
        self._generar_funcion_bu(programa)
        self.emitir("")
        # 2) Función de reconstrucción.
        self._emitir_comentario_recon()
        self.emitir(f"vector<vector<int>> {self.nombre_func}_reconstruir"
                    f"({self._params_publicos_str()}) {{")
        self.indent_level += 1
        tipo = self._obtener_tipo_vector(len(self.nombres_params))
        self.emitir(f"{tipo} tabla;")
        self._generar_assign_tabla(programa, nombre="tabla", relleno="0")
        self._generar_llenado(programa)
        self._emitir_reconstruccion(programa)
        self.indent_level -= 1
        self.emitir("}")

    def _generar_clase_recon(self, programa: ProgramaDP) -> None:
        self.emitir(f"class {self.nombre_func} {{")
        self.indent_level += 1
        for d in programa.declaraciones:
            self.emitir(f"{d.tipo.to_cpp()} {d.nombre};")
        tipo = self._obtener_tipo_vector(len(self.nombres_params))
        self.emitir(f"{tipo} tabla;")
        self.emitir("")
        self.emitir("void llenar() {")
        self.indent_level += 1
        self._generar_llenado(programa)
        self.indent_level -= 1
        self.emitir("}")
        self.indent_level -= 1
        self.emitir("")
        self.emitir("public:")
        self.indent_level += 1
        ctor_params, init_list = self._firma_constructor(programa)
        self.emitir(f"{self.nombre_func}({ctor_params}){init_list} {{")
        self.indent_level += 1
        self._generar_assign_tabla(programa, nombre="tabla", relleno="0")
        self.emitir("llenar();")
        self.indent_level -= 1
        self.emitir("}")
        self.emitir("")
        op_params = ", ".join(f"int {p}" for p in self.nombres_params)
        acceso = "tabla" + "".join(f"[{p}]" for p in self.nombres_params)
        self.emitir(f"int operator()({op_params}) {{ return {acceso}; }}")
        self.emitir("")
        self._emitir_comentario_recon()
        self.emitir("vector<vector<int>> reconstruir() {")
        self.indent_level += 1
        self._emitir_reconstruccion(programa)
        self.indent_level -= 1
        self.emitir("}")
        self.indent_level -= 1
        self.emitir("};")

    # --- Descenso que sigue la decisión óptima ----------------------------

    def _emitir_comentario_recon(self) -> None:
        if self._es_arbol:
            self.emitir("// Reconstrucción (DP de intervalos): parentización óptima como")
            self.emitir("// lista de cortes {i, j, k} (el intervalo [i, j] se parte en k),")
            self.emitir("// en preorden del árbol de decisiones.")
        else:
            self.emitir("// Reconstrucción: secuencia de estados (celdas) del camino óptimo,")
            self.emitir("// desde la llamada inicial hasta el caso base.")

    def _emitir_reconstruccion(self, programa: ProgramaDP) -> None:
        """Emite el vector de salida, la lambda recursiva que sigue la decisión
        óptima (recomputada sobre la tabla) y la llamada inicial. Para un DP de
        camino la recursión es lineal (una secuencia de celdas); para un DP de
        intervalos se ramifica en dos (una parentización)."""
        salida = "cortes" if self._es_arbol else "camino"
        self.emitir(f"vector<vector<int>> {salida};")
        firma = "void(" + ", ".join("int" for _ in self.nombres_params) + ")"
        params = ", ".join(f"int {p}" for p in self.nombres_params)
        self.emitir(f"function<{firma}> rec = [&]({params}) {{")
        self.indent_level += 1
        if self._es_arbol:
            self._emitir_cuerpo_lambda_arbol(programa)
        else:
            self._emitir_cuerpo_lambda_camino(programa)
        self.indent_level -= 1
        self.emitir("};")
        inits = ", ".join(self.visit_Expresion(a) for a in programa.retorno.argumentos)
        self.emitir(f"rec({inits});")
        self.emitir(f"return {salida};")

    def _emitir_cuerpo_lambda_camino(self, programa: ProgramaDP) -> None:
        """Cuerpo de la lambda para un DP de camino: registra la celda actual,
        para en el caso base y, en otro caso, sigue al único subproblema que
        realiza el óptimo."""
        acceso = "tabla" + "".join(f"[{p}]" for p in self.nombres_params)
        celda = "{" + ", ".join(self.nombres_params) + "}"
        self.emitir(f"camino.push_back({celda});")
        disy = self._disyuncion_base(programa)
        if disy:
            self.emitir(f"if ({disy}) return;")
        for eq in programa.ecuaciones:
            if not eq.es_caso_base:
                self._emitir_rama_lambda(eq, acceso, arbol=False)

    def _emitir_cuerpo_lambda_arbol(self, programa: ProgramaDP) -> None:
        """Cuerpo de la lambda para un DP de intervalos: para en el caso base
        (un intervalo unitario, sin corte) y, en otro caso, registra el corte k
        que realiza el óptimo y desciende en los dos subintervalos."""
        acceso = "tabla" + "".join(f"[{p}]" for p in self.nombres_params)
        disy = self._disyuncion_base(programa)
        if disy:
            self.emitir(f"if ({disy}) return;")
        for eq in programa.ecuaciones:
            if not eq.es_caso_base:
                self._emitir_rama_lambda(eq, acceso, arbol=True)

    def _emitir_rama_lambda(self, eq: Ecuacion, acceso: str, arbol: bool) -> None:
        """Para una ecuación recursiva: si su guarda se cumple, busca la
        alternativa (o el valor del iterador) cuyo valor recomputado iguala
        `tabla[celda]` y desciende a su(s) subproblema(s)."""
        conds = self._condiciones_implicitas(eq)
        if eq.condicion is not None:
            conds.append(self.visit_Expresion(eq.condicion))
        guarda = " && ".join(conds) if conds else None
        if guarda is not None:
            self.emitir(f"if ({guarda}) {{")
            self.indent_level += 1

        red = self._encontrar_reduccion(eq.der)
        if red is not None and red.rango is not None:
            # Reducción con rango: bucle sobre el iterador buscando el argóptimo.
            alt = self._sustituir_nodo(eq.der, red, red.argumentos[0])
            it = red.rango.iterador.nombre
            lo = self.visit_Expresion(red.rango.limite_inf)
            hi = self.visit_Expresion(red.rango.limite_sup)
            ini = lo if red.rango.incluye_inf else f"{lo} + 1"
            cmp = "<=" if red.rango.incluye_sup else "<"
            self.emitir(f"for (int {it} = {ini}; {it} {cmp} {hi}; {it}++) {{")
            self.indent_level += 1
            cond_match = f"{acceso} == {self.visit_Expresion(alt)}"
            if red.filtro is not None:
                # Solo los k que pasan el filtro pudieron realizar el óptimo.
                cond_match = f"({self.visit_Expresion(red.filtro)}) && ({cond_match})"
            self.emitir(f"if ({cond_match}) {{")
            self.indent_level += 1
            self._emitir_paso(alt, it, arbol)
            self.emitir("return;")
            self.indent_level -= 1
            self.emitir("}")
            self.indent_level -= 1
            self.emitir("}")
        else:
            for alt in self._alternativas(eq):
                self.emitir(f"if ({acceso} == {self.visit_Expresion(alt)}) {{")
                self.indent_level += 1
                self._emitir_paso(alt, None, arbol)
                self.emitir("return;")
                self.indent_level -= 1
                self.emitir("}")

        if guarda is not None:
            self.indent_level -= 1
            self.emitir("}")

    def _emitir_paso(self, alt: Expresion, iterador: Optional[str], arbol: bool) -> None:
        """Registra la decisión y desciende. En un DP de camino hay un único
        subproblema (la recursión es lineal); en uno de intervalos se registra
        el corte {i, j, k} y se desciende en los dos subintervalos."""
        llamadas = self._llamadas_dp(alt)
        if arbol:
            fila = "{" + ", ".join(self.nombres_params + [iterador]) + "}"
            self.emitir(f"cortes.push_back({fila});")
        for call in llamadas:
            args = ", ".join(self.visit_Expresion(a) for a in call.argumentos)
            self.emitir(f"rec({args});")

    def _disyuncion_base(self, programa: ProgramaDP) -> str:
        """Disyunción de las condiciones de los casos base (para detener el
        descenso). Cadena vacía si no hay casos base con condición."""
        disyuntos = []
        for eq in programa.ecuaciones:
            if not eq.es_caso_base:
                continue
            conds = self._condiciones_implicitas(eq)
            if eq.condicion is not None:
                conds.append(self.visit_Expresion(eq.condicion))
            disyuntos.append("(" + " && ".join(conds) + ")" if conds else "true")
        return " || ".join(disyuntos)
