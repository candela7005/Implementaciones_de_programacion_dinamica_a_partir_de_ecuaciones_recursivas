#!/usr/bin/env python3
"""Lanza toda la suite de tests (compilador + C++ generado).

Uso:
    python run_tests.py

Equivale a `python -m unittest discover -s tests -v`, pero con un único punto de
entrada. Devuelve código de salida 0 si todo pasa, 1 en caso contrario.
"""

import os
import sys
import unittest

AQUI = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.join(AQUI, "tests"))
    resultado = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if resultado.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
