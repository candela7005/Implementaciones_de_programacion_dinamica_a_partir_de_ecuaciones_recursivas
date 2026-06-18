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

@dataclass
class ProgramaDP:
    declaraciones: List[Declaracion] = field(default_factory=list)
    ecuaciones: List[Ecuacion] = field(default_factory=list)
    retorno: Llamada = None
    precondiciones: List[Precondicion] = field(default_factory=list)




def inferir_tamanos_tabla(programa: ProgramaDP) -> List[str]:
    """Tamaño (símbolo o literal) de cada dimensión de la tabla, inferido de
    los argumentos de la llamada de retorno y de las dimensiones globales."""
    tamanos: List[str] = []
    limites_globales = [
        decl.nombre for decl in programa.declaraciones
        if isinstance(decl.tipo, BasicType) and decl.tipo.clase in (BasicKind.NAT, BasicKind.INT)
    ]
    for i, arg in enumerate(programa.retorno.argumentos):
        if isinstance(arg, Variable):
            tamanos.append(arg.nombre)
        elif i < len(limites_globales): # asumimos que el orden de los argumentos coincide con el de las declaraciones globales
            tamanos.append(limites_globales[i])
        elif limites_globales:
            tamanos.append(limites_globales[-1])
        else:
            tamanos.append("100")
    return tamanos
