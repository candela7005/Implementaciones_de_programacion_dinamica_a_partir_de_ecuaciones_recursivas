"""Análisis de terminación (función de cota y lema de Farkas) y verificación
de índices fuera de rango, con un verificador propio y otro basado en el SMT Z3."""
from typing import List, Optional, Dict, Iterable, Tuple
from modelo import *


# ===========================================================================
# Análisis de terminación: función de cota + obligaciones de prueba
# ===========================================================================
# Reemplaza las heurísticas de la versión anterior (`_verifica_convergencia_
# relacional`, `obtener_modificacion`) por un esquema formal:
#
#   - Se enuncia una función de cota   μ : (params formales) → ℕ
#   - Para cada llamada recursiva en el cuerpo de cada ecuación se genera
#     una obligación: bajo las hipótesis del caso (condiciones implícitas
#     del LHS, condición explícita del `if`, negación de las condiciones
#     de los casos anteriores y restricciones del rango si la llamada está
#     dentro de una reducción min/max), se debe cumplir
#         μ(actuales) - μ(recursivos) ≥ 1     (decrecimiento estricto)
#         μ(actuales) ≥ 0                     (acotación inferior)
#   - Las obligaciones se descargan con un solver lineal propio
#     (Fourier–Motzkin reducido + búsqueda de combinación no negativa).
#
# Las clases viven aquí porque no dependen de SemanticChecks ni del codegen.

@dataclass
class LinearExpr:
    """Expresión afín entera: const + Σ coefᵢ · varᵢ.

    Las variables se identifican por su nombre. Una expresión es 'constante'
    cuando ``coefs`` está vacío.
    """
    coefs: Dict[str, int] = field(default_factory=dict)
    const: int = 0

    @classmethod
    def cero(cls) -> "LinearExpr":
        return cls()

    @classmethod
    def constante(cls, n: int) -> "LinearExpr":
        return cls(const=int(n))

    @classmethod
    def variable(cls, nombre: str) -> "LinearExpr":
        return cls(coefs={nombre: 1})

    def es_constante(self) -> bool:
        return not self.coefs

    def __add__(self, other: "LinearExpr") -> "LinearExpr":
        nuevos = dict(self.coefs)
        for v, c in other.coefs.items():
            nuevos[v] = nuevos.get(v, 0) + c
            if nuevos[v] == 0:
                del nuevos[v]
        return LinearExpr(coefs=nuevos, const=self.const + other.const)

    def __sub__(self, other: "LinearExpr") -> "LinearExpr":
        return self + other.escalado(-1)

    def __neg__(self) -> "LinearExpr":
        return self.escalado(-1)

    def escalado(self, k: int) -> "LinearExpr":
        if k == 0:
            return LinearExpr.cero()
        return LinearExpr(
            coefs={v: c * k for v, c in self.coefs.items()},
            const=self.const * k,
        )

    def __str__(self) -> str:
        if self.es_constante():
            return str(self.const)
        partes = []
        for v in sorted(self.coefs):
            c = self.coefs[v]
            signo = "+" if c > 0 else "-"
            mag = abs(c)
            partes.append(f"{signo} {v}" if mag == 1 else f"{signo} {mag}*{v}")
        if self.const != 0:
            signo = "+" if self.const > 0 else "-"
            partes.append(f"{signo} {abs(self.const)}")
        s = " ".join(partes)
        return s.lstrip("+ ").strip() if s.startswith("+ ") else s


def expr_a_lineal(nodo: Expresion, sustitucion: Optional[Dict[str, str]] = None) -> Optional[LinearExpr]:
    """Convierte una expresión del AST a `LinearExpr` si es lineal entera.

    Devuelve ``None`` si contiene divisiones, productos no triviales,
    accesos a array, llamadas, reducciones u otra estructura no lineal.

    ``sustitucion`` permite renombrar variables del cuerpo (p. ej. cuando
    el LHS introduce un nombre local distinto del formal).
    """
    sus = sustitucion or {}
    match nodo:
        case Numero(valor=v):
            return LinearExpr.constante(int(v))
        case Variable(nombre=n, indices=idxs) if not idxs:
            return LinearExpr.variable(sus.get(n, n))
        case OperacionBinaria(izq=izq, operador=op, der=der):
            l = expr_a_lineal(izq, sus)
            r = expr_a_lineal(der, sus)
            if l is None or r is None:
                return None
            if op == "+": return l + r
            if op == "-": return l - r
            if op == "*":
                if l.es_constante():
                    return r.escalado(l.const)
                if r.es_constante():
                    return l.escalado(r.const)
                return None  # producto no lineal
            return None  # división, etc.
        case _:
            return None


@dataclass
class Restriccion:
    """Restricción lineal en forma canónica ``expr OP 0`` con ``OP ∈ {>=, ==}``.

    Las desigualdades estrictas ``>`` se modelan como ``expr - 1 >= 0``
    (sobre los enteros), y las ``<=`` se invierten (``-expr >= 0``), de
    forma que el solver solo trate dos casos. Las restricciones ``!=``
    no son representables y simplemente se descartan.
    """
    expr: LinearExpr
    op: str  # '>=' | '=='

    def __str__(self) -> str:
        return f"{self.expr} {self.op} 0"


def _restricciones_de_atomo(op: str, izq, der, sus: Optional[Dict[str, str]] = None) -> List[Restriccion]:
    """Convierte un átomo relacional ``izq OP der`` a una lista de Restricciones."""
    le = expr_a_lineal(izq, sus)
    re = expr_a_lineal(der, sus)
    if le is None or re is None:
        return []
    diff = le - re
    if op == "==":
        return [Restriccion(diff, "==")]
    if op == "!=":
        return []  # no representable como conjunción lineal
    if op == ">=":
        return [Restriccion(diff, ">=")]
    if op == ">":
        return [Restriccion(diff - LinearExpr.constante(1), ">=")]
    if op == "<=":
        return [Restriccion(-diff, ">=")]
    if op == "<":
        return [Restriccion(-diff - LinearExpr.constante(1), ">=")]
    return []


def restricciones_de_condicion(cond, sus: Optional[Dict[str, str]] = None) -> List[Restriccion]:
    """Extrae las restricciones lineales que se pueden deducir de una condición.

    Las partes no lineales o no representables (``or``, ``!=`` sobre tipos
    distintos de nat, comparaciones con accesos a arrays, etc.) se descartan
    silenciosamente: la pérdida de información puede impedir demostrar una
    obligación, pero nunca produce una conclusión incorrecta.
    """
    if cond is None:
        return []
    match cond:
        case OperacionBinaria(operador="and", izq=izq, der=der):
            return restricciones_de_condicion(izq, sus) + restricciones_de_condicion(der, sus)
        case OperacionBinaria(operador="or"):
            return []  # disyunción: no la convertimos a conjunción
        case OperacionBinaria(operador=op, izq=izq, der=der) \
                if op in ("<", "<=", ">", ">=", "==", "!="):
            return _restricciones_de_atomo(op, izq, der, sus)
        case _:
            return []


def negar_atomo(op: str, izq, der, sus: Optional[Dict[str, str]] = None) -> List[Restriccion]:
    """Negación lineal-segura de un átomo. ``a == b`` se descarta porque su
    negación es disyuntiva; el resto se invierte directamente."""
    inversos = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "!=": "=="}
    if op == "==":
        return []  # negación es '!=' (no representable como conjunción lineal)
    if op in inversos:
        return _restricciones_de_atomo(inversos[op], izq, der, sus)
    return []


def negar_condicion(cond, sus: Optional[Dict[str, str]] = None) -> List[Restriccion]:
    """Negación segura de una condición compuesta. Para ``A and B`` la
    negación es ``¬A or ¬B`` (disyuntiva), así que devolvemos lista vacía.
    Para un único átomo, usamos `negar_atomo`.
    """
    if cond is None:
        return []
    match cond:
        case OperacionBinaria(operador="and"):
            return []
        case OperacionBinaria(operador="or"):
            return []
        case OperacionBinaria(operador=op, izq=izq, der=der) \
                if op in ("<", "<=", ">", ">=", "==", "!="):
            return negar_atomo(op, izq, der, sus)
        case _:
            return []


@dataclass
class Obligacion:
    """Obligación ``hipotesis ⊨ goal``.

    ``goal`` es ``meta_expr >= 0``. Para el decrecimiento estricto:
    ``meta_expr = μ(actuales) - μ(recursivos) - 1``. Para la acotación:
    ``meta_expr = μ(actuales)``.
    """
    descripcion: str
    hipotesis: List[Restriccion]
    meta_expr: LinearExpr  # debe demostrarse >= 0
    nombre: str            # 'decrecimiento' | 'cota >= 0'


# ---------------------------------------------------------------------------
# Recolectora
# ---------------------------------------------------------------------------

class TerminacionRecolectora:
    """Construye la lista de obligaciones a partir del AST del programa.

    El recorrido es independiente de la elección de μ; eso lo decide el
    cliente (típicamente `analizar_terminacion`) probando varios candidatos.
    """

    def __init__(self, programa: ProgramaDP):
        self.programa = programa
        self.nombre_func = programa.retorno.nombre
        self.parametros = self._inferir_parametros()

    def _inferir_parametros(self) -> List[str]:
        """Nombres formales: los de la ecuación con MÁS nombres de variable
        DISTINTOS en su LHS (ver nota en CodeGenerator._inferir_params_formales)."""
        nombres: List[str] = []
        mejor_distintos = -1
        for eq in self.programa.ecuaciones:
            vs = [a.nombre for a in eq.izq.argumentos if isinstance(a, Variable)]
            distintos = len(set(vs))
            if distintos > mejor_distintos:
                mejor_distintos = distintos
                nombres = vs
        if not nombres:
            # fallback (casos con todos los args literales: poco habitual)
            n = len(self.programa.ecuaciones[0].izq.argumentos) if self.programa.ecuaciones else 0
            nombres = [f"_p{i}" for i in range(n)]
        return nombres

    # -- núcleo ------------------------------------------------------------

    def _restricciones_lhs(self, eq: Ecuacion) -> Tuple[List[Restriccion], Dict[str, str]]:
        """Restricciones implícitas a partir del patrón del LHS, junto con un
        mapa de renombrado (nombre local → formal) para los Variables que
        usen un nombre distinto al formal en esa posición.
        """
        restricciones: List[Restriccion] = []
        renombrado: Dict[str, str] = {}
        for i, arg in enumerate(eq.izq.argumentos):
            if i >= len(self.parametros):
                continue
            formal = self.parametros[i]
            match arg:
                case Numero(valor=v):
                    # formal == v
                    restricciones.append(Restriccion(
                        LinearExpr.variable(formal) - LinearExpr.constante(int(v)),
                        "==",
                    ))
                case Variable(nombre=n, indices=idxs) if not idxs:
                    if n in self.parametros and n != formal:
                        # f(i, i): el segundo formal coincide con el primero
                        restricciones.append(Restriccion(
                            LinearExpr.variable(formal) - LinearExpr.variable(n),
                            "==",
                        ))
                    elif n != formal:
                        # nombre local que renombra el formal
                        renombrado[n] = formal
        return restricciones, renombrado

    def _hipotesis_negacion_anteriores(
        self,
        anteriores: List[Tuple[List[Restriccion], Optional[Expresion], Dict[str, str]]],
    ) -> List[Restriccion]:
        """Para cada caso anterior, intenta negar UNA de sus restricciones
        sin disyunciones. Si no es posible para ese caso entero, lo descarta.
        """
        resultado: List[Restriccion] = []
        # Constantes a las que cada parámetro queda fijado por algún caso base
        # anterior (patrón  p == c  en el LHS).
        consts_por_param: Dict[str, set] = {}

        for restr_lhs, cond_expr, _ren in anteriores:
            if len(restr_lhs) == 1 and restr_lhs[0].op == "==":
                e = restr_lhs[0].expr  # forma  p - c == 0
                if len(e.coefs) == 1 and list(e.coefs.values())[0] == 1:
                    p = next(iter(e.coefs))
                    consts_por_param.setdefault(p, set()).add(-e.const)  # p == c
            elif cond_expr is not None and not restr_lhs:
                # Caso con solo condición explícita: si es un único átomo lineal,
                # su negación sí es expresable (p. ej. ¬(i > 1) ≡ i ≤ 1).
                resultado.extend(negar_condicion(cond_expr))

        # La negación de ¬(p==c) es disyuntiva, PERO la conjunción de todas las
        # exclusiones es expresable: si los casos base cubren {0,1,…,t}, entonces
        # (sobre nat) p ∉ {0,…,t} ∧ p ≥ 0  ⟹  p ≥ t+1.
        for p, consts in consts_por_param.items():
            t = -1
            while (t + 1) in consts:
                t += 1
            if t >= 0:
                resultado.append(
                    Restriccion(LinearExpr.variable(p) - LinearExpr.constante(t + 1), ">=")
                )
        return resultado

    def _llamadas_recursivas(
        self,
        nodo: Expresion,
        rangos_activos: List[Rango],
        sus: Dict[str, str],
    ) -> Iterable[Tuple[Llamada, List[Restriccion]]]:
        """Itera (llamada_recursiva, hipótesis_extra_por_estar_en_rangos).

        Las llamadas dentro de un `min{i ≤ k < j}(...)` heredan como hipótesis
        las desigualdades del rango (lim_inf ≤ k, k op lim_sup).
        """
        match nodo:
            case Llamada(nombre=n) if n == self.nombre_func:
                yield nodo, self._hipotesis_de_rangos(rangos_activos, sus)
            case OperacionBinaria(izq=izq, der=der):
                yield from self._llamadas_recursivas(izq, rangos_activos, sus)
                yield from self._llamadas_recursivas(der, rangos_activos, sus)
            case Reduccion(rango=rg, argumentos=args):
                nuevos = rangos_activos + ([rg] if rg is not None else [])
                for a in args:
                    yield from self._llamadas_recursivas(a, nuevos, sus)
            case Llamada(argumentos=args):  # llamadas no recursivas (raras): seguimos
                for a in args:
                    yield from self._llamadas_recursivas(a, rangos_activos, sus)
            case _:
                return

    def _hipotesis_de_rangos(self, rangos: List[Rango], sus: Dict[str, str]) -> List[Restriccion]:
        hyps: List[Restriccion] = []
        for r in rangos:
            it = sus.get(r.iterador.nombre, r.iterador.nombre)
            inf = expr_a_lineal(r.limite_inf, sus)
            sup = expr_a_lineal(r.limite_sup, sus)
            if inf is not None:
                if r.incluye_inf:  # lim_inf <= it  ↔  it - lim_inf >= 0
                    hyps.append(Restriccion(LinearExpr.variable(it) - inf, ">="))
                else:               # lim_inf < it   ↔  it - lim_inf - 1 >= 0
                    hyps.append(Restriccion(LinearExpr.variable(it) - inf - LinearExpr.constante(1), ">="))
            if sup is not None:
                if r.incluye_sup:  # it <= sup  ↔  sup - it >= 0
                    hyps.append(Restriccion(sup - LinearExpr.variable(it), ">="))
                else:               # it < sup   ↔  sup - it - 1 >= 0
                    hyps.append(Restriccion(sup - LinearExpr.variable(it) - LinearExpr.constante(1), ">="))
        return hyps

    def _hipotesis_tipos_nat(self) -> List[Restriccion]:
        """Cada parámetro formal nat aporta la hipótesis ``formal >= 0``."""
        return [Restriccion(LinearExpr.variable(p), ">=") for p in self.parametros]

    # -- API ---------------------------------------------------------------

    def recolectar(self, mu: LinearExpr) -> List[Obligacion]:
        """Construye las obligaciones suponiendo la cota ``mu`` (LinearExpr
        sobre los nombres formales en ``self.parametros``).
        """
        obligaciones: List[Obligacion] = []
        anteriores: List[Tuple[List[Restriccion], Optional[Expresion], Dict[str, str]]] = []

        hyps_globales = self._hipotesis_tipos_nat()

        for idx, eq in enumerate(self.programa.ecuaciones):
            restr_lhs, renombrado = self._restricciones_lhs(eq)
            cond_explicita = restricciones_de_condicion(eq.condicion, renombrado)
            hyps_negaciones = self._hipotesis_negacion_anteriores(anteriores)

            hyps_caso = hyps_globales + restr_lhs + cond_explicita + hyps_negaciones

            for llamada_rec, hyps_rango in self._llamadas_recursivas(eq.der, [], renombrado):
                hyps_total = hyps_caso + hyps_rango
                # μ(actuales): los parámetros formales mismos (sin renombrar:
                # μ está expresada sobre los formales).
                mu_actual = mu
                # μ(rec): solo necesitamos linealizar los argumentos cuyo formal
                # aparece en μ. Para los demás, da igual lo que valgan.
                mu_rec = LinearExpr.constante(mu.const)
                args_descripcion: List[str] = []
                no_lineal_relevante = False
                for pos, formal in enumerate(self.parametros):
                    arg = llamada_rec.argumentos[pos]
                    if formal in mu.coefs:
                        arg_lineal = expr_a_lineal(arg, renombrado)
                        if arg_lineal is None:
                            no_lineal_relevante = True
                            break
                        mu_rec = mu_rec + arg_lineal.escalado(mu.coefs[formal])
                        args_descripcion.append(str(arg_lineal))
                    else:
                        # Parámetro irrelevante para μ: para diagnóstico, lo describimos
                        # como expresión literal si es lineal, o '?' en caso contrario.
                        a_lin = expr_a_lineal(arg, renombrado)
                        args_descripcion.append(str(a_lin) if a_lin is not None else "?")

                if no_lineal_relevante:
                    # Este candidato no puede analizarse: marcamos un fallo "soft"
                    # añadiendo una obligación trivialmente no demostrable.
                    obligaciones.append(Obligacion(
                        descripcion=(
                            f"caso {idx} ({_pp_lhs(eq.izq, self.parametros)})  →  "
                            f"argumento recursivo no lineal en parámetro relevante"
                        ),
                        hipotesis=[],
                        meta_expr=LinearExpr.constante(-1),  # nunca >= 0
                        nombre="decrecimiento (no lineal)",
                    ))
                    continue

                desc = (
                    f"caso {idx} ({_pp_lhs(eq.izq, self.parametros)})  →  "
                    f"llamada {self.nombre_func}({', '.join(args_descripcion)})"
                )

                # Decrecimiento estricto: μ(actuales) - μ(rec) - 1 >= 0
                obligaciones.append(Obligacion(
                    descripcion=desc,
                    hipotesis=hyps_total,
                    meta_expr=(mu_actual - mu_rec) - LinearExpr.constante(1),
                    nombre="decrecimiento",
                ))
                # Acotación: μ(actuales) >= 0
                obligaciones.append(Obligacion(
                    descripcion=desc,
                    hipotesis=hyps_total,
                    meta_expr=mu_actual,
                    nombre="cota >= 0",
                ))

            anteriores.append((restr_lhs, eq.condicion, renombrado))

        return obligaciones


def _sustituir_lineal(expr: LinearExpr, mapa: Dict[str, LinearExpr]) -> LinearExpr:
    """Sustituye cada variable en ``expr`` por la expresión lineal de ``mapa``.
    Las variables no presentes en el mapa se conservan tal cual."""
    res = LinearExpr.constante(expr.const)
    for v, c in expr.coefs.items():
        if v in mapa:
            res = res + mapa[v].escalado(c)
        else:
            res = res + LinearExpr.variable(v).escalado(c)
    return res


def _pp_expr(nodo) -> str:
    """Pretty-print compacto de una expresión para mensajes de diagnóstico."""
    match nodo:
        case Numero(valor=v):
            return str(v)
        case Variable(nombre=n, indices=idxs):
            return n + "".join(f"[{_pp_expr(i)}]" for i in idxs)
        case OperacionBinaria(izq=izq, operador=op, der=der):
            return f"{_pp_expr(izq)} {op} {_pp_expr(der)}"
        case Llamada(nombre=nm, argumentos=args):
            return f"{nm}({', '.join(_pp_expr(a) for a in args)})"
        case Reduccion(tipo=t, argumentos=args):
            return f"{t}{{…}}({', '.join(_pp_expr(a) for a in args)})"
        case _:
            return "?"


def _pp_lhs(llamada: Llamada, parametros: List[str]) -> str:
    """Pretty-print del LHS de una ecuación para mensajes de diagnóstico."""
    return f"{llamada.nombre}({', '.join(_pp_expr(a) for a in llamada.argumentos)})"


# ---------------------------------------------------------------------------
# Verificadora
# ---------------------------------------------------------------------------

class TerminacionVerificadora:
    """Solver lineal sencillo para descargar obligaciones del fragmento

        meta_expr ≥ 0  bajo  conjunción de  ``eᵢ >= 0``  y  ``eⱼ == 0``.

    Estrategia:
      1. Sustituir las igualdades para eliminar variables (una a una).
      2. Combinar las desigualdades restantes buscando coeficientes
         no negativos que produzcan ``meta_expr - k`` con ``k ≥ 0`` y
         coefs todos cero (Farkas constructivo, búsqueda finita en
         {0,1}^|hyps| ya es suficiente para los ejemplos tratados).
      3. Si todo lo anterior falla, intentar un caso especial: la meta
         es directamente una constante no negativa tras simplificar.
    """

    def demostrar(self, obl: Obligacion) -> bool:
        meta = obl.meta_expr
        igualdades: List[LinearExpr] = []
        desigualdades: List[LinearExpr] = []
        for h in obl.hipotesis:
            if h.op == "==":
                igualdades.append(h.expr)
            else:  # '>='
                desigualdades.append(h.expr)

        # 1) Eliminar variables usando igualdades  e == 0  →  v = -resto/c
        meta, desigualdades = self._eliminar_igualdades(meta, igualdades, desigualdades)

        # 2) ¿La meta es ya constante no negativa?
        if meta.es_constante() and meta.const >= 0:
            return True

        # 3) Buscar combinación no negativa de desigualdades que produzca meta
        if self._meta_es_combinacion_no_negativa(meta, desigualdades):
            return True

        return False

    # -- pasos -------------------------------------------------------------

    def _eliminar_igualdades(
        self,
        meta: LinearExpr,
        igualdades: List[LinearExpr],
        desigualdades: List[LinearExpr],
    ) -> Tuple[LinearExpr, List[LinearExpr]]:
        """Resuelve las igualdades para alguna variable y sustituye.

        Si una igualdad ``e == 0`` tiene una variable con coeficiente ±1,
        despejamos esa variable y la sustituimos en meta y desigualdades.
        Repetimos hasta no poder más.
        """
        cambiado = True
        while cambiado and igualdades:
            cambiado = False
            for k, eq in enumerate(igualdades):
                # Buscar una variable con coeficiente ±1
                for v, c in eq.coefs.items():
                    if c in (1, -1):
                        # eq == 0  →  c·v + resto == 0  →  v = -resto/c
                        resto = LinearExpr(
                            coefs={vv: cc for vv, cc in eq.coefs.items() if vv != v},
                            const=eq.const,
                        )
                        valor_v = resto.escalado(-c)  # -resto/c con c=±1
                        sub = {v: valor_v}
                        meta = _sustituir_lineal(meta, sub)
                        desigualdades = [_sustituir_lineal(d, sub) for d in desigualdades]
                        igualdades = (
                            [_sustituir_lineal(e, sub) for j, e in enumerate(igualdades) if j != k]
                        )
                        cambiado = True
                        break
                if cambiado:
                    break
        return meta, desigualdades

    def _meta_es_combinacion_no_negativa(
        self,
        meta: LinearExpr,
        desigualdades: List[LinearExpr],
    ) -> bool:
        """¿Existen λᵢ ∈ ℕ tales que  meta = Σ λᵢ · desigᵢ + k  con k ≥ 0?

        Búsqueda en {0,1,2}^n para n pequeño (las obligaciones reales casi
        siempre se descargan con un único término).
        """
        n = len(desigualdades)
        if n == 0:
            return meta.es_constante() and meta.const >= 0

        # Enumerar combinaciones λ ∈ {0,1,2}^n. Con n ≤ ~6 es trivial.
        # Si n es mayor, probamos solo subconjuntos pequeños (≤ 3 hipótesis activas).
        from itertools import product
        max_lambda = 2 if n <= 6 else 1
        for lams in product(range(max_lambda + 1), repeat=n):
            comb = LinearExpr.cero()
            for lam, d in zip(lams, desigualdades):
                if lam:
                    comb = comb + d.escalado(lam)
            resto = meta - comb
            if resto.es_constante() and resto.const >= 0:
                return True
        return False


# ---------------------------------------------------------------------------
# Generador de candidatos de cota y orquestador
# ---------------------------------------------------------------------------

def _candidatos_cota(parametros: List[str]) -> List[LinearExpr]:
    """Lista ordenada de candidatos μ a probar."""
    candidatos: List[LinearExpr] = []
    # 1. Cada parámetro individual
    for p in parametros:
        candidatos.append(LinearExpr.variable(p))
    # 2. Suma de todos los parámetros
    if len(parametros) >= 2:
        suma = LinearExpr.cero()
        for p in parametros:
            suma = suma + LinearExpr.variable(p)
        candidatos.append(suma)
    # 3. Diferencia entre pares (para problemas tipo secMatrices: j - i)
    for i, pi in enumerate(parametros):
        for j, pj in enumerate(parametros):
            if i != j:
                candidatos.append(LinearExpr.variable(pj) - LinearExpr.variable(pi))
    return candidatos


def analizar_terminacion(programa: ProgramaDP, verificadora=None) -> Tuple[LinearExpr, List[Obligacion]]:
    """Punto de entrada. Prueba candidatos de cota; el primero que cumple
    todas las obligaciones se devuelve junto con sus obligaciones.

    ``verificadora`` permite inyectar un descargador de obligaciones alternativo
    (p. ej. ``VerificadoraSMT`` con Z3). Por defecto usa el solver propio.

    Lanza ``ValueError`` con detalle si ningún candidato funciona.
    """
    recolectora = TerminacionRecolectora(programa)
    if verificadora is None:
        verificadora = TerminacionVerificadora()

    # Si no hay ecuaciones recursivas: terminación trivial.
    hay_recursividad = any(
        any(_contiene_llamada_a(eq.der, recolectora.nombre_func) for eq in [eq])
        for eq in programa.ecuaciones
    )
    if not hay_recursividad:
        return LinearExpr.constante(0), []

    # Si hay precondiciones y se pidió el SMT, se usa la vía con Z3 (teoría de
    # arrays + cuantificadores), capaz de aprovechar precondiciones como
    # `forall k: moneda[k] >= 1` que el solver lineal propio no puede usar.
    if programa.precondiciones and isinstance(verificadora, VerificadoraSMT) and Z3_DISPONIBLE:
        return _terminacion_con_precondiciones_z3(programa, recolectora)

    fallos_por_candidato: List[Tuple[LinearExpr, List[Obligacion]]] = []
    for mu in _candidatos_cota(recolectora.parametros):
        obligaciones = recolectora.recolectar(mu)
        no_demostradas = [o for o in obligaciones if not verificadora.demostrar(o)]
        if not no_demostradas:
            return mu, obligaciones
        fallos_por_candidato.append((mu, no_demostradas))

    # Diagnóstico: tomar el candidato con menos fallos
    fallos_por_candidato.sort(key=lambda par: len(par[1]))
    mu_mejor, fallos = fallos_por_candidato[0]
    detalles = "\n  ".join(
        f"[{o.nombre}]  {o.descripcion}\n    meta: {o.meta_expr} >= 0\n    "
        f"hipótesis: {', '.join(str(h) for h in o.hipotesis) or '(ninguna)'}"
        for o in fallos[:3]
    )
    raise ValueError(
        f"Error de Terminación: no se encontró una función de cota válida.\n"
        f"Mejor candidato probado: μ = {mu_mejor}\n"
        f"Obligaciones no demostradas (hasta 3):\n  {detalles}"
    )


def _contiene_llamada_a(nodo, nombre: str) -> bool:
    match nodo:
        case Llamada(nombre=n) if n == nombre:
            return True
        case OperacionBinaria(izq=izq, der=der):
            return _contiene_llamada_a(izq, nombre) or _contiene_llamada_a(der, nombre)
        case Reduccion(argumentos=args):
            return any(_contiene_llamada_a(a, nombre) for a in args)
        case Llamada(argumentos=args):
            return any(_contiene_llamada_a(a, nombre) for a in args)
        case _:
            return False




# ===========================================================================
# Verificación alternativa con un SMT (Z3): terminación e índices en rango
# ===========================================================================
# El uso de un SMT es OPCIONAL (bandera --smt). Ofrece una segunda vía,
# independiente del solver propio, para:
#   (a) descargar las mismas obligaciones de terminación;
#   (b) comprobar, mediante un invariante inductivo de acotación, que los
#       índices de la tabla y de los arrays no se salen de rango.
# Si la librería z3 no está instalada, las funciones lo señalan con claridad.

try:
    import z3 as _z3
    Z3_DISPONIBLE = True
except ImportError:  # pragma: no cover - depende del entorno
    _z3 = None
    Z3_DISPONIBLE = False

# Límite de tiempo por consulta a Z3 (en milisegundos). La aritmética no lineal
# con cuantificadores (arrays + forall) es indecidible en general, así que una
# consulta podría no terminar; al agotar el límite, Z3 devuelve `unknown`. Por
# la semántica de las comprobaciones esto NUNCA produce un falso positivo: en
# índices (refutar) un `unknown` no marca nada, y en terminación significa "no
# demostrado" (rechazo conservador). Se aplica siempre que se use Z3.
SMT_TIMEOUT_MS = 10000


def _lineal_a_z3(expr: LinearExpr, cache: dict):
    """Traduce una expresión lineal a un término entero de Z3."""
    termino = _z3.IntVal(expr.const)
    for v, c in expr.coefs.items():
        if v not in cache:
            cache[v] = _z3.Int(v)
        termino = termino + c * cache[v]
    return termino


def _restriccion_a_z3(r: Restriccion, cache: dict):
    e = _lineal_a_z3(r.expr, cache)
    return e == 0 if r.op == "==" else e >= 0


def _smt_demuestra(hipotesis: List[Restriccion], meta: LinearExpr) -> bool:
    """¿Es válida la implicación  (⋀ hipótesis) → (meta ≥ 0)?

    Equivale a comprobar que  (⋀ hipótesis) ∧ (meta < 0)  es INSATISFACIBLE.
    """
    solver = _z3.Solver()
    solver.set("timeout", SMT_TIMEOUT_MS)
    cache: dict = {}
    for h in hipotesis:
        solver.add(_restriccion_a_z3(h, cache))
    solver.add(_lineal_a_z3(meta, cache) < 0)
    return solver.check() == _z3.unsat


class VerificadoraSMT:
    """Descargador de obligaciones de terminación que delega en Z3. Es un
    sustituto directo de `TerminacionVerificadora` (mismo método `demostrar`)."""

    def demostrar(self, obl: Obligacion) -> bool:
        return _smt_demuestra(obl.hipotesis, obl.meta_expr)


# ---------------------------------------------------------------------------
# Terminación con precondiciones (cláusula `requires`): teoría de arrays + ∀
# ---------------------------------------------------------------------------
# Esta vía traduce directamente el AST a Z3 (con arrays y cuantificadores), de
# modo que puede usar precondiciones como `forall k: moneda[k] >= 1` para probar
# el decrecimiento de recurrencias dependientes de los datos (p. ej. el cambio
# de monedas), algo que el solver lineal propio no puede hacer. Solo se activa
# con --smt (y si hay precondiciones).

def _array_z3(nombre: str, num_indices: int):
    """Crea (o reutiliza) una constante de Z3 de tipo array de `num_indices`
    dimensiones de enteros a enteros, p. ej. moneda : Array(Int, Int)."""
    sort = _z3.IntSort()
    for _ in range(num_indices):
        sort = _z3.ArraySort(_z3.IntSort(), sort)
    return _z3.Const(nombre, sort)


def _ast_a_z3(nodo, ints: dict, arrays: dict, ren: Optional[Dict[str, str]] = None):
    """Traduce una expresión del AST a un término/fórmula de Z3, modelando los
    accesos a arrays con la teoría de arrays (Select). ``ren`` renombra las
    variables escalares (nombre local → formal), igual que ``expr_a_lineal``,
    para que casen con las hipótesis lineales ya expresadas en los formales."""
    ren = ren or {}
    match nodo:
        case Numero(valor=v):
            return _z3.IntVal(v)
        case Variable(nombre=n, indices=idxs) if not idxs:
            n = ren.get(n, n)
            if n not in ints:
                ints[n] = _z3.Int(n)
            return ints[n]
        case Variable(nombre=n, indices=idxs):
            if n not in arrays:
                arrays[n] = _array_z3(n, len(idxs))
            termino = arrays[n]
            for idx in idxs:
                termino = _z3.Select(termino, _ast_a_z3(idx, ints, arrays, ren))
            return termino
        case OperacionBinaria(izq=izq, operador=op, der=der):
            a = _ast_a_z3(izq, ints, arrays, ren)
            b = _ast_a_z3(der, ints, arrays, ren)
            return {
                "+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b,
                "<": lambda: a < b, "<=": lambda: a <= b,
                ">": lambda: a > b, ">=": lambda: a >= b,
                "==": lambda: a == b, "!=": lambda: a != b,
                "and": lambda: _z3.And(a, b), "or": lambda: _z3.Or(a, b),
            }[op]()
        case _:
            raise ValueError(f"No se puede traducir a Z3: {type(nodo).__name__}")


def _precondicion_a_z3(pc: Precondicion, ints: dict, arrays: dict):
    """Traduce una precondición a una fórmula de Z3. La forma `forall k: φ`
    se traduce como ForAll([k], φ) usando un nombre fresco para no colisionar
    con las variables libres de la obligación."""
    if pc.cuantificador is None:
        return _ast_a_z3(pc.expr, ints, arrays)
    var = pc.cuantificador
    ligada = _z3.Int("_q_" + var)
    previo = ints.get(var)
    ints[var] = ligada                       # vincula k → variable ligada
    cuerpo = _ast_a_z3(pc.expr, ints, arrays)
    if previo is None:
        del ints[var]
    else:
        ints[var] = previo
    return _z3.ForAll([ligada], cuerpo)


def _z3_valida(hipotesis, meta_z3) -> bool:
    """¿Es válida  (⋀ hipótesis) → (meta_z3 ≥ 0)?  ⟺  hipótesis ∧ meta_z3<0 UNSAT."""
    s = _z3.Solver()
    s.set("timeout", SMT_TIMEOUT_MS)
    for h in hipotesis:
        s.add(h)
    s.add(meta_z3 < 0)
    return s.check() == _z3.unsat


def _z3_refuta(hipotesis, meta_z3) -> bool:
    """¿Existe un CONTRAEJEMPLO concreto de  meta_z3 ≥ 0?  Es decir, ¿es
    (⋀ hipótesis) ∧ meta_z3 < 0 SATISFACIBLE?

    A diferencia de ``_z3_valida``, solo devuelve True ante un modelo definido
    (``sat``); ante ``unknown`` (p. ej. por los cuantificadores) devuelve False.
    Así, la verificación de índices nunca señala un acceso salvo que Z3 exhiba
    una entrada real que lo viole: incompleta, pero SIN falsos positivos."""
    s = _z3.Solver()
    s.set("timeout", SMT_TIMEOUT_MS)
    for h in hipotesis:
        s.add(h)
    s.add(meta_z3 < 0)
    return s.check() == _z3.sat


def _terminacion_con_precondiciones_z3(programa: ProgramaDP, rec: "TerminacionRecolectora"):
    """Análisis de terminación vía Z3, aprovechando las precondiciones. Prueba
    los mismos candidatos de cota y, por cada llamada recursiva, comprueba en
    Z3 el decrecimiento y la acotación, con las precondiciones como hipótesis."""
    for mu in _candidatos_cota(rec.parametros):
        if _mu_valida_z3(rec, mu, programa.precondiciones):
            return mu, []
    raise ValueError(
        "Error de Terminación: no se encontró una función de cota válida "
        "(con SMT y las precondiciones dadas)."
    )


def _mu_valida_z3(rec: "TerminacionRecolectora", mu: LinearExpr,
                  precondiciones: List[Precondicion]) -> bool:
    """¿Cumple μ el decrecimiento y la acotación en todas las llamadas, usando
    Z3 y las precondiciones?"""
    anteriores = []
    nat = rec._hipotesis_tipos_nat()
    for eq in rec.programa.ecuaciones:
        restr_lhs, ren = rec._restricciones_lhs(eq)
        cond = restricciones_de_condicion(eq.condicion, ren)
        neg = rec._hipotesis_negacion_anteriores(anteriores)
        hyps_lin = nat + restr_lhs + cond + neg
        for llamada, hyps_rg in rec._llamadas_recursivas(eq.der, [], ren):
            if not _comprueba_llamada_z3(rec, mu, llamada, hyps_lin + hyps_rg, eq, precondiciones):
                return False
        anteriores.append((restr_lhs, eq.condicion, ren))
    return True


def _comprueba_llamada_z3(rec, mu, llamada, hyps_lin, eq, precondiciones) -> bool:
    ints: dict = {}
    arrays: dict = {}
    hyps = [_restriccion_a_z3(h, ints) for h in hyps_lin]
    # La condición explícita completa (puede tener arrays, p. ej. moneda[i] <= v)
    if eq.condicion is not None:
        hyps.append(_ast_a_z3(eq.condicion, ints, arrays))
    # Las precondiciones (con arrays y cuantificadores)
    for pc in precondiciones:
        hyps.append(_precondicion_a_z3(pc, ints, arrays))

    # μ(actual) sobre los parámetros formales y μ(siguiente) sustituyendo por
    # los argumentos de la llamada recursiva.
    formales_como_args = [Variable(p) for p in rec.parametros]
    mu_actual = _mu_a_z3(mu, rec.parametros, formales_como_args, ints, arrays)
    mu_sig = _mu_a_z3(mu, rec.parametros, llamada.argumentos, ints, arrays)

    decrece = _z3_valida(hyps, (mu_actual - mu_sig) - 1)   # μ_act − μ_sig − 1 ≥ 0
    acotada = _z3_valida(hyps, mu_actual)                  # μ_act ≥ 0
    return decrece and acotada


def _mu_a_z3(mu: LinearExpr, parametros: List[str], args, ints: dict, arrays: dict):
    """Evalúa μ (lineal sobre los formales) sustituyendo cada formal por el
    término Z3 del argumento correspondiente."""
    termino = _z3.IntVal(mu.const)
    for p, c in mu.coefs.items():
        pos = parametros.index(p)
        termino = termino + c * _ast_a_z3(args[pos], ints, arrays)
    return termino


class VerificadorIndices:
    """Comprueba que los índices no se salen de rango.

    Usa un INVARIANTE INDUCTIVO de acotación: suponiendo que la celda actual
    (x₁,…,x_d) está dentro de la tabla (0 ≤ xₖ ≤ Sₖ) y las hipótesis del caso,
    demuestra que toda celda leída por la recurrencia también lo está. Para los
    accesos a arrays comprueba la cota inferior (índice ≥ 0), que es la que
    provoca los accesos fuera de rango por la izquierda (p. ej. `tabla[-1]`).

    Las obligaciones (`hipótesis lineales ⊨ índice ≥ 0`) son del mismo tipo que
    las de terminación, así que se descargan con CUALQUIER verificador: el
    solver propio (`TerminacionVerificadora`, por defecto) o Z3
    (`VerificadoraSMT`). No depende de SMT.

    El análisis es SOUND pero incompleto: si no logra demostrar una cota, lo
    reporta como aviso (no como error), porque podría faltar una precondición
    sobre los datos (caso típico: `d[i-1]` del producto de matrices, donde
    i ≥ 1 proviene de la forma de la llamada inicial).
    """

    def __init__(self, programa: ProgramaDP, verificadora=None):
        self.programa = programa
        self.verificadora = verificadora if verificadora is not None else TerminacionVerificadora()
        self.rec = TerminacionRecolectora(programa)
        self.parametros = self.rec.parametros
        self.tamanos = inferir_tamanos_tabla(programa)
        self.arrays = {
            d.nombre for d in programa.declaraciones if isinstance(d.tipo, ArrayType)
        }
        self.cotas_inf = self._cotas_inferiores()  # cota inferior inductiva por parámetro
        self.precondiciones = programa.precondiciones
        # Bajo --smt (verificadora Z3) se intentan demostrar TAMBIÉN los índices
        # NO lineales (dependientes de los datos, p. ej. c - w[i]): requiere
        # teoría de arrays y, si las hay, las precondiciones. El solver propio no
        # entra en esto (no representa arrays), así que con él se omiten.
        self._smt_arrays = (
            Z3_DISPONIBLE
            and isinstance(self.verificadora, VerificadoraSMT)
        )

    def _prueba(self, hipotesis: List[Restriccion], meta: LinearExpr) -> bool:
        """Descarga la obligación  (⋀ hipótesis) ⊨ (meta ≥ 0)  con el verificador
        configurado (solver propio o SMT)."""
        obl = Obligacion(descripcion="índice", hipotesis=hipotesis, meta_expr=meta, nombre="índice")
        return self.verificadora.demostrar(obl)

    # -- hipótesis ---------------------------------------------------------

    def _size_expr(self, k: int) -> Optional[LinearExpr]:
        if k >= len(self.tamanos):
            return None
        s = self.tamanos[k]
        return LinearExpr.constante(int(s)) if s.lstrip("-").isdigit() else LinearExpr.variable(s)

    def _invariante(self) -> List[Restriccion]:
        """Lₖ ≤ xₖ ≤ Sₖ para cada parámetro, donde Lₖ es la cota inferior
        inductiva inferida (≥ 0 por defecto, mayor si la recurrencia la
        mantiene; p. ej. i ≥ 1 en el producto de matrices)."""
        hyps: List[Restriccion] = []
        for k, p in enumerate(self.parametros):
            lk = self.cotas_inf.get(p, 0)
            hyps.append(Restriccion(LinearExpr.variable(p) - LinearExpr.constante(lk), ">="))  # xₖ ≥ Lₖ
            s = self._size_expr(k)
            if s is not None:
                hyps.append(Restriccion(s - LinearExpr.variable(p), ">="))  # Sₖ - xₖ ≥ 0
        return hyps

    def _cotas_inferiores(self) -> Dict[str, int]:
        """Infiere una cota inferior constante Lₖ por parámetro tal que xₖ ≥ Lₖ
        sea un INVARIANTE INDUCTIVO: vale en la llamada inicial y se preserva en
        toda llamada recursiva. Parte de las constantes de la llamada inicial
        (p. ej. secMatrices(1, N) sugiere i ≥ 1) y baja a 0 cualquier cota cuya
        preservación no pueda demostrarse. Es siempre SOUND: en el peor caso
        queda Lₖ = 0 (la cota de los naturales)."""
        ret_args = self.programa.retorno.argumentos
        L: Dict[str, int] = {}
        for k, p in enumerate(self.parametros):
            arg = ret_args[k] if k < len(ret_args) else None
            L[p] = arg.valor if isinstance(arg, Numero) else 0

        def invariante_local() -> List[Restriccion]:
            return [Restriccion(LinearExpr.variable(p) - LinearExpr.constante(L[p]), ">=")
                    for p in self.parametros]

        cambiado = True
        while cambiado:
            cambiado = False
            anteriores: List[Tuple[List[Restriccion], Optional[Expresion], Dict[str, str]]] = []
            for eq in self.programa.ecuaciones:
                restr_lhs, ren = self.rec._restricciones_lhs(eq)
                base = (restr_lhs
                        + restricciones_de_condicion(eq.condicion, ren)
                        + self.rec._hipotesis_negacion_anteriores(anteriores)
                        + invariante_local())
                for llamada, hyps_rg in self.rec._llamadas_recursivas(eq.der, [], ren):
                    for k, p in enumerate(self.parametros):
                        if L[p] == 0 or k >= len(llamada.argumentos):
                            continue
                        arg_lin = expr_a_lineal(llamada.argumentos[k], ren)
                        meta = None if arg_lin is None else arg_lin - LinearExpr.constante(L[p])
                        if meta is None or not self._prueba(base + hyps_rg, meta):
                            L[p] = 0  # no se preserva: caer a la cota nat (sound)
                            cambiado = True
                anteriores.append((restr_lhs, eq.condicion, ren))
        return L

    # -- recorrido de accesos a arrays ------------------------------------

    def _accesos_array(self, nodo, rangos: List[Rango], sus: Dict[str, str]):
        """Itera (lista_de_índices, hipótesis_de_rango) por cada acceso a un
        array declarado."""
        match nodo:
            case Variable(nombre=n, indices=idxs) if idxs and n in self.arrays:
                yield list(idxs), self.rec._hipotesis_de_rangos(rangos, sus)
                for ix in idxs:
                    yield from self._accesos_array(ix, rangos, sus)
            case Variable(indices=idxs):
                for ix in idxs:
                    yield from self._accesos_array(ix, rangos, sus)
            case OperacionBinaria(izq=izq, der=der):
                yield from self._accesos_array(izq, rangos, sus)
                yield from self._accesos_array(der, rangos, sus)
            case Llamada(argumentos=args):
                for a in args:
                    yield from self._accesos_array(a, rangos, sus)
            case Reduccion(rango=rg, argumentos=args):
                nuevos = rangos + ([rg] if rg is not None else [])
                if rg is not None:
                    yield from self._accesos_array(rg.limite_inf, rangos, sus)
                    yield from self._accesos_array(rg.limite_sup, rangos, sus)
                for a in args:
                    yield from self._accesos_array(a, nuevos, sus)
            case _:
                return

    # -- análisis ----------------------------------------------------------

    def analizar(self) -> List[str]:
        """Devuelve la lista de avisos (vacía si todo se demuestra en rango)."""
        avisos: List[str] = []
        invariante = self._invariante()
        anteriores: List[Tuple[List[Restriccion], Optional[Expresion], Dict[str, str]]] = []

        for eq in self.programa.ecuaciones:
            restr_lhs, ren = self.rec._restricciones_lhs(eq)
            cond = restricciones_de_condicion(eq.condicion, ren)
            neg = self.rec._hipotesis_negacion_anteriores(anteriores)
            hyps_caso = invariante + restr_lhs + cond + neg

            # (a) Índices de la tabla: cada llamada recursiva debe caer en rango.
            for llamada, hyps_rg in self.rec._llamadas_recursivas(eq.der, [], ren):
                self._chequear_llamada(llamada, hyps_caso + hyps_rg, ren, avisos, eq.condicion)

            # (b) Índices de arrays: cota inferior (≥ 0).
            for indices, hyps_rg in self._accesos_array(eq.der, [], ren):
                for idx in indices:
                    self._chequear_array(idx, hyps_caso + hyps_rg, ren, avisos, eq.condicion)

            anteriores.append((restr_lhs, eq.condicion, ren))

        return avisos

    def _chequear_llamada(self, llamada: Llamada, hyps, ren, avisos: List[str],
                          guarda=None) -> None:
        for k, arg in enumerate(llamada.argumentos):
            e = expr_a_lineal(arg, ren)
            if e is None:
                # Índice no lineal (depende de los datos, p. ej. c - w[i]). Bajo
                # --smt se intenta refutar con Z3 (teoría de arrays); con el solver
                # propio queda fuera del fragmento decidible y se omite.
                if self._smt_arrays:
                    viol = self._refutar_indice_z3(arg, hyps, ren, guarda, cota_sup_k=k)
                    if viol:
                        avisos.append(
                            f"no se pudo demostrar que el índice {k} de "
                            f"{_pp_lhs(llamada, self.parametros)} sea {viol}"
                        )
                continue
            # cota inferior  e ≥ 0
            if not self._prueba(hyps, e):
                avisos.append(
                    f"no se pudo demostrar que el índice {k} de "
                    f"{_pp_lhs(llamada, self.parametros)} sea ≥ 0"
                )
            # cota superior  Sₖ - e ≥ 0
            s = self._size_expr(k)
            if s is not None and not self._prueba(hyps, s - e):
                avisos.append(
                    f"no se pudo demostrar que el índice {k} de "
                    f"{_pp_lhs(llamada, self.parametros)} sea ≤ {self.tamanos[k]}"
                )

    def _chequear_array(self, idx, hyps, ren, avisos: List[str], guarda=None) -> None:
        e = expr_a_lineal(idx, ren)
        if e is None:
            if self._smt_arrays:
                viol = self._refutar_indice_z3(idx, hyps, ren, guarda, cota_sup_k=None)
                if viol:
                    avisos.append(f"no se pudo demostrar que un índice de array sea {viol}")
            return
        if not self._prueba(hyps, e):
            avisos.append(f"no se pudo demostrar que el índice de array '{e}' sea ≥ 0")

    # -- comprobación de índices NO lineales con Z3 (teoría de arrays + ∀) ---

    @staticmethod
    def _dims_base(tipo):
        """Número de dimensiones de un tipo array y su tipo base de elemento."""
        n = 0
        while isinstance(tipo, ArrayType):
            n += 1
            tipo = tipo.elemento
        return n, tipo

    def _hyps_arrays_nat_z3(self, arrays: dict):
        """Asume arr[·] ≥ 0 para cada array de naturales declarado. Es SOUND
        (los `nat` lo son) y evita contraejemplos espurios con elementos de
        array negativos, que de otro modo invalidarían cotas superiores
        legítimas (p. ej. v - moneda[i] ≤ V con moneda[i] ≥ 0)."""
        extra = []
        for d in self.programa.declaraciones:
            if not isinstance(d.tipo, ArrayType):
                continue
            n, base = self._dims_base(d.tipo)
            if not (isinstance(base, BasicType) and base.clase == BasicKind.NAT):
                continue
            arr = _array_z3(d.nombre, n)
            arrays[d.nombre] = arr  # fija el sort antes de traducir los accesos
            poss = [_z3.Int(f"_p_{d.nombre}_{j}") for j in range(n)]
            sel = arr
            for p in poss:
                sel = _z3.Select(sel, p)
            extra.append(_z3.ForAll(poss, sel >= 0))
        return extra

    def _refutar_indice_z3(self, arg, hyps_lin, ren, guarda, cota_sup_k):
        """Devuelve un texto ("≥ 0" / "≤ S") si Z3 exhibe un contraejemplo de
        que el índice `arg` esté en rango, o None si no puede refutarlo. Reúne
        como hipótesis: la no negatividad de los arrays nat, las hipótesis
        lineales del caso, la guarda completa del `if` (con su parte no lineal,
        p. ej. moneda[i] ≤ v) y las precondiciones (`requires`)."""
        try:
            ints: dict = {}
            arrays: dict = {}
            hyps = self._hyps_arrays_nat_z3(arrays)
            hyps += [_restriccion_a_z3(h, ints) for h in hyps_lin]
            if guarda is not None:
                hyps.append(_ast_a_z3(guarda, ints, arrays, ren))
            for pc in self.precondiciones:
                hyps.append(_precondicion_a_z3(pc, ints, arrays))
            idx = _ast_a_z3(arg, ints, arrays, ren)
        except (ValueError, KeyError):
            return None  # no traducible a Z3: se omite (sin falsos positivos)
        # cota inferior  idx ≥ 0
        if _z3_refuta(hyps, idx):
            return "≥ 0"
        # cota superior  Sₖ - idx ≥ 0
        if cota_sup_k is not None:
            s = self._size_expr(cota_sup_k)
            if s is not None and _z3_refuta(hyps, _lineal_a_z3(s, ints) - idx):
                return f"≤ {self.tamanos[cota_sup_k]}"
        return None
