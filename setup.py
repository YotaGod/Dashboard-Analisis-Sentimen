# setup.py
import subprocess
import sys

print("🔄 Menginstall semua dependencies...")
print("=" * 60)

packages = [
    "tensorflow==2.16.1",
    "protobuf==4.25.3",
    "streamlit==1.32.0",
    "numpy==1.26.4",
    "pandas==2.2.1",
    "matplotlib==3.8.3",
    "Sastrawi==1.0.1",
    "nltk==3.8.1",
    "h5py==3.10.0"
]

# Install semua package
subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages + ["--no-cache-dir"])

print("\n✅ Instalasi selesai!")
print("\n🔍 Melakukan verifikasi...")
print("=" * 60)

# Verifikasi
import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import Sastrawi
import nltk
import h5py
from google.protobuf import __version__ as protobuf_version

# Keras sudah include dalam tensorflow 2.16.1
import keras

print(f"✅ TensorFlow   : {tf.__version__}")
print(f"✅ Keras        : {keras.__version__}")
print(f"✅ Protobuf     : {protobuf_version}")
print(f"✅ Streamlit    : {st.__version__}")
print(f"✅ NumPy        : {np.__version__}")
print(f"✅ Pandas       : {pd.__version__}")
print(f"✅ Matplotlib   : {matplotlib.__version__}")
print(f"✅ Sastrawi     : 1.0.1 (installed)")
print(f"✅ NLTK         : {nltk.__version__}")
print(f"✅ h5py         : {h5py.__version__}")
print(f"✅ Python       : {sys.version.split()[0]}")
print("=" * 60)
print("🎉 SEMUA DEPENDENCIES SUDAH SIAP!")
print("=" * 60)