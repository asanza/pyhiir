#!/usr/bin/env python3
"""Launcher — run from the repo root: python pyhiir_ui.py"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.pyhiir.ui.app import main
main()
