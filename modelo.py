"""Modelo de datos del compilador: sistema de tipos y árbol de sintaxis
abstracta (AST), junto con utilidades sobre el programa."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class BasicKind(Enum):
    NAT  = "nat"
    INT  = "int"
    REAL = "real"
    BOOL = "bool"
    CHAR = "char"

    @classmethod
    def desde_lexema(cls, lex: str) -> "BasicKind":
        return cls(lex)


class Type:
    """Raíz de la jerarquía de tipos del DSL."""

    def to_cpp(self) -> str:
        raise NotImplementedError

    def es_numerico(self) -> bool:
        return False


@dataclass
class BasicType(Type):
    clase: BasicKind

    def to_cpp(self) -> str:
        # nat se promueve a int en C++; real → double; el resto coinciden.
        match self.clase:
            case BasicKind.NAT | BasicKind.INT:
                return "int"
            case BasicKind.REAL:
                return "double"
            case BasicKind.BOOL:
                return "bool"
            case BasicKind.CHAR:
                return "char"

    def es_numerico(self) -> bool:
        return self.clase in (BasicKind.NAT, BasicKind.INT, BasicKind.REAL)


@dataclass
class ArrayType(Type):
    elemento: Type
    dim_nombre: Optional[str] = None  # tamaño nombrado opcional, p.ej. array<nat, N>

    def to_cpp(self) -> str:
        return f"vector<{self.elemento.to_cpp()}>"


class Expresion:
    pass

@dataclass
class Numero(Expresion):
    valor: int

@dataclass
class Variable(Expresion):
    nombre: str
    indices: List[Expresion] = field(default_factory=list) # Para cosas como w[i] o d[i-1]
    tipo: Optional[Type] = None

@dataclass
class OperacionBinaria(Expresion):
    izq: Expresion
    operador: str        
    der: Expresion

@dataclass
class Rango:
    limite_inf: Expresion
    iterador: Variable        # 'k'
    limite_sup: Expresion
    incluye_sup: bool         # True si el límite superior usa '<=', False si '<'
    incluye_inf: bool = True  # True si el límite inferior usa '<=', False si '<'

@dataclass
class Reduccion(Expresion):
    tipo: str            # 'min', 'max' o 'sum'
    rango: Optional[Rango] # min{i <= k < j}(...)
    argumentos: List[Expresion] # por ej (a, b)
    filtro: Optional[Expresion] = None  # condición opcional sobre el iterador: min{i<=k<j : phi}(...)

@dataclass
class Declaracion:
    tipo: Type
    nombre: str

@dataclass
class Llamada(Expresion):
    nombre: str
    argumentos: List[Expresion] 

@dataclass
class Ecuacion:
    izq: Llamada
    der: Expresion
    condicion: Optional[Expresion] = None
    es_caso_base: bool = False

@dataclass
class Precondicion:
    """Precondición sobre los datos de entrada (cláusula `requires`).
    `cuantificador` es el nombre de la variable universal si la precondición es
    de la forma `forall k: φ`, o None si es una condición simple `φ`."""
    expr: Expresion
    cuantificador: Optional[str] = None

def _buscar_llamada(nodo) -> Optional[Llamada]:
    """Primera llamada que aparece en una expresión (búsqueda en profundidad)."""
    match nodo:
        case Llamada():
            return nodo
        case OperacionBinaria(izq=izq, der=der):
            return _buscar_llamada(izq) or _buscar_llamada(der)
        case Reduccion(argumentos=args):
            for a in args:
                r = _buscar_llamada(a)
                if r is not None:
                    return r
            return None
        case _:
            return None


def _sustituir_variable(nodo, nombre: str, reemplazo):
    """Copia de `nodo` con cada Variable `nombre` (sin índices) sustituida por
    `reemplazo`. Sirve para representar la celda extrema de un retorno agregado
    (el iterador de la reducción pasa a ser el límite superior del rango)."""
    match nodo:
        case Variable(nombre=n, indices=idxs) if n == nombre and not idxs:
            return reemplazo
        case Variable(nombre=n, indices=idxs):
            return Variable(n, [_sustituir_variable(i, nombre, reemplazo) for i in idxs])
        case OperacionBinaria(izq=l, operador=op, der=r):
            return OperacionBinaria(_sustituir_variable(l, nombre, reemplazo), op,
                                    _sustituir_variable(r, nombre, reemplazo))
        case Llamada(nombre=fn, argumentos=args):
            return Llamada(fn, [_sustituir_variable(a, nombre, reemplazo) for a in args])
        case _:
            return nodo


def llamada_representativa(retorno) -> Llamada:
    """La llamada a la función DP que representa el retorno, para nombrar la
    función, dimensionar la tabla y comprobar la llamada inicial.

    - Si el retorno es una llamada `f(args)`, ella misma.
    - Si es un retorno AGREGADO `max/min{a <= k <= b}( … f(…) … )` (la LIS
      global, por ejemplo), la llamada interna con el iterador `k` sustituido
      por el límite superior `b`: la celda extrema, que fija el nombre, las
      dimensiones de la tabla y la cota de la llamada inicial."""
    if isinstance(retorno, Llamada):
        return retorno
    if isinstance(retorno, Reduccion):
        interna = None
        for a in retorno.argumentos:
            interna = _buscar_llamada(a)
            if interna is not None:
                break
        if interna is None:
            raise ValueError("[Semántico] El retorno agregado no llama a la función.")
        if retorno.rango is not None:
            interna = _sustituir_variable(
                interna, retorno.rango.iterador.nombre, retorno.rango.limite_sup)
        return interna
    raise ValueError("[Semántico] El retorno debe ser una llamada f(...) o una "
                     "reducción max/min de una llamada.")


@dataclass
class ProgramaDP:
    declaraciones: List[Declaracion] = field(default_factory=list)
    ecuaciones: List[Ecuacion] = field(default_factory=list)
    retorno: Expresion = None          # Llamada f(...) o Reduccion (retorno agregado)
    precondiciones: List[Precondicion] = field(default_factory=list)

    @property
    def llamada_inicial(self) -> Llamada:
        """Llamada DP representativa del retorno (ver `llamada_representativa`)."""
        return llamada_representativa(self.retorno)

    @property
    def retorno_agregado(self) -> bool:
        """True si el retorno es una reducción max/min sobre las celdas."""
        return isinstance(self.retorno, Reduccion)


def inferir_tamanos_tabla(programa: ProgramaDP) -> List[str]:
    """Tamaño (símbolo o literal) de cada dimensión de la tabla, inferido de
    los argumentos de la llamada de retorno (representativa) y de las
    dimensiones globales."""
    tamanos: List[str] = []
    limites_globales = [
        decl.nombre for decl in programa.declaraciones
        if isinstance(decl.tipo, BasicType) and decl.tipo.clase in (BasicKind.NAT, BasicKind.INT)
    ]
    for i, arg in enumerate(programa.llamada_inicial.argumentos):
        if isinstance(arg, Variable):
            tamanos.append(arg.nombre)
        elif i < len(limites_globales): # asumimos que el orden de los argumentos coincide con el de las declaraciones globales
            tamanos.append(limites_globales[i])
        elif limites_globales:
            tamanos.append(limites_globales[-1])
        else:
            tamanos.append("100")
    return tamanos
