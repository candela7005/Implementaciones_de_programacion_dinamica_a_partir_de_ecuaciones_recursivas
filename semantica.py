"""Análisis semántico: comprobación de tipos, declaraciones y consistencia,
e invocación del análisis de terminación."""
from typing import Dict, Optional
from modelo import *
from terminacion import LinearExpr, analizar_terminacion, VerificadoraSMT


class SemanticChecks:
    """Comprobaciones semánticas del programa: tipos, declaraciones y
    consistencia del nombre de la función. La prueba de terminación se
    delega al módulo de análisis basado en función de cota
    (`analizar_terminacion`), invocado al final de `validar_programa`.
    """

    def __init__(self):
        self.globales: Dict[str, Type] = {}
        self.locales: Dict[str, Type] = {}
        self.func: Optional[str] = None
        self.cota_inferida: Optional[LinearExpr] = None  # se rellena tras analizar_terminacion

    def validar_programa(self, programa: ProgramaDP, usar_smt: bool = False) -> None:
        """Punto de entrada para validar todo el programa. Si ``usar_smt``,
        la prueba de terminación se descarga con Z3 en vez del solver propio."""
        for decl in programa.declaraciones:
            self.declaracion(decl)

        for pre in programa.precondiciones:
            self.precondicion(pre)

        for eq in programa.ecuaciones:
            self.ecuacion(eq)

        self.inicial(programa.retorno)

        # Prueba de terminación basada en función de cota (Fase 2 del plan).
        verificadora = VerificadoraSMT() if usar_smt else None
        self.cota_inferida, _obligaciones = analizar_terminacion(programa, verificadora)

    def comprobacionFunc(self, nombre_func: str, contexto: str = "") -> None:
        if self.func is None:
            self.func = nombre_func
        elif nombre_func != self.func:
            raise ValueError(
                f"[Semántico] Función distinta '{nombre_func}' en {contexto}; "
                f"esperada: '{self.func}'"
            )

    def declaracion(self, decl: Declaracion) -> None:
        if decl.nombre in self.globales:
            raise ValueError(f"[Semántico] Identificador duplicado: '{decl.nombre}'")
        self.globales[decl.nombre] = decl.tipo

    def precondicion(self, pre) -> None:
        """Valida una precondición `requires`. La variable cuantificada (si la
        hay) se registra como `nat` local mientras se comprueba el cuerpo."""
        self.locales.clear()
        if pre.cuantificador is not None:
            self.locales[pre.cuantificador] = BasicType(BasicKind.NAT)
        tipo = self.validar_y_anotar(pre.expr)
        if not (isinstance(tipo, BasicType) and tipo.clase == BasicKind.BOOL):
            raise ValueError("[Semántico] Una precondición 'requires' debe ser una condición booleana.")

    def validar_y_anotar(self, nodo) -> Type:
        """
        Recorre el árbol, valida que todo exista y anota los nodos Variable
        con su tipo real para la futura generación de C++. Devuelve el `Type`
        inferido para el nodo.
        """
        match nodo:
            case Numero():
                return BasicType(BasicKind.NAT)

            case Variable():
                tipo_en_tabla = self.locales.get(nodo.nombre) or self.globales.get(nodo.nombre)
                if tipo_en_tabla is None:
                    raise ValueError(f"Error Semántico: Variable '{nodo.nombre}' no declarada.")

                tipo_actual: Type = tipo_en_tabla
                for idx in nodo.indices:
                    self.validar_y_anotar(idx)
                    if isinstance(tipo_actual, ArrayType):
                        tipo_actual = tipo_actual.elemento
                    else:
                        raise ValueError(
                            f"Error de Tipos: Se intentó indexar '{nodo.nombre}' demasiadas veces. "
                            f"No se puede indexar el tipo '{tipo_actual.to_cpp()}'."
                        )

                nodo.tipo = tipo_actual
                return tipo_actual

            case OperacionBinaria():
                t_izq = self.validar_y_anotar(nodo.izq)
                t_der = self.validar_y_anotar(nodo.der)

                # Operaciones aritméticas
                if nodo.operador in ('+', '-', '*', '/'):
                    if not (t_izq.es_numerico() and t_der.es_numerico()):
                        raise ValueError(
                            f"Error de Tipos: El operador '{nodo.operador}' requiere operandos numéricos. "
                            f"Recibió operandos no numéricos."
                        )
                    # Inferencia: real domina sobre int, int sobre nat.
                    kinds = {t_izq.clase, t_der.clase}
                    if BasicKind.REAL in kinds:
                        return BasicType(BasicKind.REAL)
                    if BasicKind.INT in kinds:
                        return BasicType(BasicKind.INT)
                    return BasicType(BasicKind.NAT)

                # Operaciones relacionales
                if nodo.operador in ('<', '<=', '>', '>=', '==', '!='):
                    son_numericos = t_izq.es_numerico() and t_der.es_numerico()
                    mismo_tipo = (
                        isinstance(t_izq, BasicType) and isinstance(t_der, BasicType)
                        and t_izq.clase == t_der.clase
                    )
                    if not (son_numericos or mismo_tipo):
                        raise ValueError(
                            f"Error de Tipos: No se pueden comparar tipos distintos no numéricos "
                            f"con el operador '{nodo.operador}'."
                        )
                    return BasicType(BasicKind.BOOL)

                # Operaciones lógicas
                if nodo.operador in ('and', 'or'):
                    # Se admiten operandos booleanos o numéricos (no-cero = verdadero),
                    # de modo que las recurrencias booleanas (p. ej. subset-sum, que
                    # combina llamadas con `or`) sean expresables. El resultado es
                    # booleano (0/1 en el C++ generado).
                    valido = lambda t: isinstance(t, BasicType) and (
                        t.clase == BasicKind.BOOL or t.es_numerico())
                    if not (valido(t_izq) and valido(t_der)):
                        raise ValueError(
                            f"Error de Tipos: El operador '{nodo.operador}' requiere "
                            f"operandos booleanos o numéricos."
                        )
                    return BasicType(BasicKind.BOOL)

                return t_izq  # fallback

            case Llamada():
                self.comprobacionFunc(nodo.nombre, "llamada")
                for i, arg in enumerate(nodo.argumentos):
                    tipo_arg = self.validar_y_anotar(arg)
                    if not tipo_arg.es_numerico():
                        raise ValueError(
                            f"Error de Tipos: El parámetro {i+1} en la llamada a '{nodo.nombre}' "
                            f"debe ser numérico."
                        )
                # Las funciones DP devuelven nat por convención
                return BasicType(BasicKind.NAT)

            case Reduccion():
                if nodo.rango:
                    self.locales[nodo.rango.iterador.nombre] = BasicType(BasicKind.NAT)
                    self.validar_y_anotar(nodo.rango.limite_inf)
                    self.validar_y_anotar(nodo.rango.limite_sup)
                    if nodo.filtro is not None:
                        t_filtro = self.validar_y_anotar(nodo.filtro)
                        if not (isinstance(t_filtro, BasicType)
                                and t_filtro.clase == BasicKind.BOOL):
                            raise ValueError(
                                "[Semántico] El filtro de una reducción (tras ':') "
                                "debe ser una condición booleana."
                            )

                for arg in nodo.argumentos:
                    self.validar_y_anotar(arg)

                return BasicType(BasicKind.NAT)

            case _:
                raise ValueError(f"Nodo de tipo no soportado en validar_y_anotar: {type(nodo).__name__}")

    def ecuacion(self, eq: Ecuacion) -> None:
        lhs = eq.izq
        rhs = eq.der
        cond = eq.condicion

        self.comprobacionFunc(lhs.nombre, "lado izquierdo de ecuación")
        self.locales.clear()

        # Registrar las variables introducidas por el LHS como locales nat.
        for param in lhs.argumentos:
            if isinstance(param, Variable):
                self.locales[param.nombre] = BasicType(BasicKind.NAT)

        if not eq.es_caso_base:
            self.validar_y_anotar(rhs)
            if cond is not None:
                self.validar_y_anotar(cond)

    def inicial(self, retorno: Llamada) -> None:
        self.comprobacionFunc(retorno.nombre, "retorno inicial")
        self.locales.clear()
        self.validar_y_anotar(retorno)
