"""Análisis léxico y sintáctico: gramática Lark, parser LALR y construcción
del AST (IRBuilder)."""
from typing import Optional
from lark import Lark, Transformer, Token
from modelo import *


GRAMMAR = r"""
start           : programa

programa        : declaraciones precondiciones ecuaciones inicial

declaraciones   : (declaracion)*
precondiciones  : (precondicion)*
ecuaciones      : (ecuacion)+
inicial         : "return" llamada ";"

declaracion     : tipo IDENT ("," IDENT)? ";"
precondicion    : "requires" pre ";"
pre             : "forall" IDENT ":" expr   -> pre_forall
                | expr                       -> pre_simple
ecuacion        : llamada "=" expr ("if" expr)? ";"

!tipo            : "nat" | "int" | "real" | "bool" | "char" | "array" "<" tipo ("," IDENT)? ">"

llamada         : IDENT "(" (expr ("," expr)*)? ")" 
reduccion       : (MIN | MAX | SUM) "{" rango (":" expr)? "}" "("expr")"
                | (MIN | MAX | SUM) "{" expr ("," expr)* "}"

rango           : (IDENT | NUMERO) op_rango IDENT op_rango (IDENT | NUMERO)
op_rango        : LT | LE

expr           : logica
logica         : cmp ((AND | OR) cmp)*
cmp            : suma ((LT | LE | GT | GE | EQ | NEQ) suma)?
suma           : producto ((PLUS | MINUS) producto)*  
producto       : primario ((MULT | DIV) primario)*
primario       : NUMERO
                | llamada
                | IDENT
                | IDENT ("[" expr "]")+
                | reduccion
                | "(" expr ")"
                
MAX: "max"
MIN: "min"
SUM: "sum"
AND: "and"
OR: "or"
LT: "<"
LE: "<="
GT: ">"
GE: ">="
EQ: "=="
NEQ: "!="
MINUS: "-"
PLUS: "+"
MULT: "*"   
DIV: "/"

%import common.CNAME -> IDENT
%import common.NUMBER -> NUMERO
%import common.WS
COMMENT: "//" /[^\n]/*
%ignore COMMENT
%ignore WS
"""

parser = Lark(GRAMMAR, parser="lalr", propagate_positions=True, maybe_placeholders=False)


# ---------------------------------------------------------------------------
# Sistema de tipos del DSL
# ---------------------------------------------------------------------------
# Representación interna estructurada para los tipos. Sustituye al manejo
# anterior basado en strings ("array<array<nat>>"), que obligaba a re-parsear
# el tipo cada vez que se quería inspeccionar o traducir a C++.


class IRBuilder(Transformer):
    
    def start(self, args): return args[0]

    def programa(self, args):
        # args = [declaraciones, precondiciones, ecuaciones, inicial]
        decls = args[0] if args[0] else []
        precs = args[1] if args[1] else []
        return ProgramaDP(declaraciones=decls, precondiciones=precs,
                          ecuaciones=args[2], retorno=args[3])

    def declaraciones(self, args):
        return [decl for sublist in args for decl in sublist]

    def precondiciones(self, args): return list(args)
    def precondicion(self, args): return args[0]
    def pre_forall(self, args):
        return Precondicion(expr=args[1], cuantificador=str(args[0].value))
    def pre_simple(self, args):
        return Precondicion(expr=args[0], cuantificador=None)

    def ecuaciones(self, args): return args

    def inicial(self, args): return args[0]

    def tipo(self, args):
        # Construye el AST de tipos directamente. Lark aplica el Transformer
        # bottom-up, así que el sub-árbol interno `tipo` ya es un objeto Type
        # cuando se procesa el caso array.
        if len(args) == 1 and isinstance(args[0], Token):
            return BasicType(clase=BasicKind.desde_lexema(args[0].value))

        elemento: Optional[Type] = None
        dim_nombre: Optional[str] = None
        for arg in args:
            if isinstance(arg, Type):
                elemento = arg
            elif isinstance(arg, Token) and arg.type == "IDENT":
                dim_nombre = str(arg.value)

        if elemento is None:
            raise ValueError("Tipo array sin tipo de elemento (gramática inconsistente).")
        return ArrayType(elemento=elemento, dim_nombre=dim_nombre)

    def declaracion(self, args):
        tipo_base = args[0] # string
        idents = args[1:]
        return [Declaracion(tipo=tipo_base, nombre=str(ident.value if isinstance(ident, Token) else ident)) for ident in idents] #TODO

    def llamada(self, args):
        nombre = str(args[0].value if isinstance(args[0], Token) else args[0])
        argumentos = args[1:] if len(args) > 1 else []
        return Llamada(nombre=nombre, argumentos=argumentos)

    def ecuacion(self, args):
        izq = args[0]
        der = args[1]
        condicion = args[2] if len(args) > 2 else None
        
        def tiene_recursividad(nodo):
            if isinstance(nodo, Llamada) and nodo.nombre == izq.nombre: return True
            if isinstance(nodo, OperacionBinaria): return tiene_recursividad(nodo.izq) or tiene_recursividad(nodo.der)
            if isinstance(nodo, Reduccion): return any(tiene_recursividad(arg) for arg in nodo.argumentos)
            if isinstance(nodo, Variable): return any(tiene_recursividad(idx) for idx in nodo.indices)
            return False

        es_base = not tiene_recursividad(der)
        return Ecuacion(izq=izq, der=der, condicion=condicion, es_caso_base=es_base)

    def reduccion(self, args):
        tipo_red = str(args[0].value).lower() # max/min/sum

        if isinstance(args[1], Rango): # tiene Rango (forma iterativa)
            rango_obj = args[1]
            # Con filtro opcional los args son [tok, rango, filtro, cuerpo];
            # sin él, [tok, rango, cuerpo].
            if len(args) == 4:
                filtro, cuerpo = args[2], args[3]
            else:
                filtro, cuerpo = None, args[2]
            return Reduccion(tipo=tipo_red, rango=rango_obj,
                             argumentos=[cuerpo], filtro=filtro)

        else:
            return Reduccion(tipo=tipo_red, rango=None, argumentos=args[1:])
    
    def logica(self, args): return self._construir_binaria(args)
    def expr(self, args): return args[0]
    def cmp(self, args):    return self._construir_binaria(args)
    def suma(self, args):   return self._construir_binaria(args)
    def producto(self, args): return self._construir_binaria(args)

    def _construir_binaria(self, args):
        if len(args) == 1: return args[0]
        nodo = args[0]
        for i in range(1, len(args), 2):
            operador = str(args[i].value if isinstance(args[i], Token) else args[i])
            nodo = OperacionBinaria(izq=nodo, operador=operador, der=args[i+1])
        return nodo

    def primario(self, args):
        if len(args) == 1:
            nodo = args[0]
            if isinstance(nodo, Token):
                if nodo.type == 'NUMERO': return Numero(int(nodo.value))
                if nodo.type == 'IDENT':  return Variable(nombre=str(nodo.value))
            return nodo
            
        if isinstance(args[0], Token) and args[0].type == 'IDENT':
            nombre = str(args[0].value)
            return Variable(nombre=nombre, indices=args[1:])
            
        return args[0]

    def op_rango(self, args):
        # Reduce el operador del rango a su string ('<' o '<=').
        return str(args[0].value)

    def rango(self, args):
        # Los límites pueden ser identificador o número literal; el iterador
        # (en medio) siempre es identificador.
        def to_expr(tok):
            if isinstance(tok, Token) and tok.type == 'NUMERO':
                return Numero(int(tok.value))
            return Variable(nombre=str(tok.value if isinstance(tok, Token) else tok))

        lim_inf = to_expr(args[0])
        iterador_var = Variable(nombre=str(args[2].value))
        lim_sup = to_expr(args[4])

        # args[1] y args[3] son los operadores (strings vía op_rango).
        incluye_inf = (args[1] == '<=')
        incluye_sup = (args[3] == '<=')

        return Rango(limite_inf=lim_inf, iterador=iterador_var, limite_sup=lim_sup,
                     incluye_sup=incluye_sup, incluye_inf=incluye_inf)
