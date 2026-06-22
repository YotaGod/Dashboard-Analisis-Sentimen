python -c "
import sys
print('=' * 60)
print('🔍 VERIFIKASI INSTALASI PACKAGE')
print('=' * 60)

packages = {
    'tensorflow': '2.16.1',
    'tf_keras': '2.16.0',
    'protobuf': '4.25.3',
    'streamlit': '1.32.0',
    'numpy': '1.26.4',
    'pandas': '2.2.1',
    'matplotlib': '3.8.3',
    'Sastrawi': '1.0.1',
    'nltk': '3.8.1',
    'h5py': '3.10.0'
}

all_ok = True

for pkg, expected_version in packages.items():
    try:
        module = __import__(pkg)
        installed_version = getattr(module, '__version__', 'Unknown')
        
        if installed_version == expected_version:
            status = '✅'
        else:
            status = '⚠️'
            all_ok = False
        
        print(f'{status} {pkg:15} : {installed_version} (expected: {expected_version})')
        
    except ImportError as e:
        print(f'❌ {pkg:15} : NOT INSTALLED')
        all_ok = False

print('=' * 60)

# Check Python version
print(f'🐍 Python Version  : {sys.version.split()[0]}')
print(f'📦 Python Path    : {sys.executable}')
print('=' * 60)

if all_ok:
    print('✅ SEMUA PACKAGE SUDAH TERINSTALL DENGAN BENAR!')
else:
    print('⚠️  ADA PACKAGE YANG PERLU DIPERBAIKI!')
    
print('=' * 60)
"