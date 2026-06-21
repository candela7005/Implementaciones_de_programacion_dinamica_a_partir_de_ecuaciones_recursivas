# Compilador de programación dinámica

Compilador de un lenguaje de dominio específico (DSL) para problemas de **programación
dinámica**. A partir de una recurrencia escrita en el DSL, el compilador la **verifica**
(que termina y que no se sale de rango) y **genera código C++** que la resuelve, en varias
estrategias.

Desarrollado como Trabajo de Fin de Grado en Matemáticas (UCM).

## Qué hace

- **Verifica** la recurrencia antes de generar nada: prueba la terminación (mediante una
  función de cota) y que los índices se mantienen en rango. Si no puede garantizarlo, la
  rechaza con un mensaje en lugar de generar código incorrecto.
- **Genera C++** en cuatro estrategias —recursión pura, top-down (memoización), bottom-up
  (tabulación) y bottom-up con optimización de espacio— y en dos estilos —funciones o
  clase—.
- Opcionalmente, **reconstruye la solución óptima** (no solo su valor) y descarga las
  comprobaciones con el demostrador **SMT Z3**.

## Requisitos

- Python 3.10 o superior
- [`lark`](https://github.com/lark-parser/lark) (analizador sintáctico): `pip install lark`
- [`z3-solver`](https://pypi.org/project/z3-solver/) (`pip install z3-solver`): necesario para
  las recurrencias con índices que dependen de los datos (como la mochila) y para la opción
  `--smt`. Sin él, esos problemas se rechazan con un mensaje en lugar de aceptarse sin verificar.
- Un compilador de C++ (g++ o MSVC) para compilar el código generado

## Uso

```
python codigoPD.py ejemplos/mochila.dp
```

Esto verifica el problema y, si es correcto, imprime el C++ generado (por defecto, estrategia
top-down y estilo de funciones). Con opciones:

```
python codigoPD.py mi_problema.dp --algoritmo bottom-up --gen clase --reconstruir -o salida.cpp
```

### Opciones

| Opción | Valores | Por defecto | Descripción |
|---|---|---|---|
| `archivo` | ruta a un `.dp` | (obligatorio) | el problema a compilar |
| `--algoritmo` | `sin-memo` / `top-down` / `bottom-up` | `top-down` | estrategia de generación |
| `--gen` | `funcion` / `clase` | `funcion` | estilo de la salida |
| `--space-opt` | (bandera) | — | optimización de espacio (solo con `bottom-up`) |
| `--reconstruir` | (bandera) | — | reconstruye también la solución óptima |
| `--smt` | (bandera) | — | usa Z3 con las precondiciones al probar la terminación (p. ej. el cambio de monedas). Los índices que dependen de los datos ya se verifican con Z3 sin este flag |
| `--out`, `-o` | fichero | stdout | dónde escribir el C++ |

### El DSL

Una recurrencia se escribe como un sistema de ecuaciones. Por ejemplo, la mochila 0/1:

```
nat N, W;
array<nat> v, w;
mochila(0, c) = 0;
mochila(i, 0) = 0;
mochila(i, c) = mochila(i - 1, c) if w[i] > c;
mochila(i, c) = max{ mochila(i-1, c), v[i] + mochila(i-1, c - w[i]) } if w[i] <= c;
return mochila(N, W);
```

Hay más ejemplos en [`ejemplos/`](ejemplos): subsecuencia común más larga, distancia de
edición, producto de matrices, corte de varilla, cambio de monedas, subsecuencia creciente
más larga, etc.

> El código generado es la función (o clase) que resuelve el problema, **sin `main`**. Para
> ejecutarlo, añade un `main` que proporcione los datos y llame a la función. Los arrays son
> **base-1**: el elemento k está en la posición k y la posición 0 es un centinela.

## Pruebas

```
python run_tests.py
```

Genera el C++ de cada ejemplo, lo compila y lo ejecuta en todas las estrategias y estilos,
comprueba que el resultado es el esperado, y verifica que las recurrencias mal planteadas se
rechazan. (Requiere un compilador de C++.)

## Estructura

| Fichero | Contenido |
|---|---|
| `codigoPD.py` | interfaz de línea de comandos |
| `sintaxis.py` | gramática y construcción del AST |
| `modelo.py` | tipos y AST |
| `semantica.py` | comprobaciones de tipos |
| `terminacion.py` | verificación de terminación e índices |
| `generacion.py` | generación de código C++ |
| `ejemplos/` | problemas de ejemplo |
| `tests/` | pruebas |
