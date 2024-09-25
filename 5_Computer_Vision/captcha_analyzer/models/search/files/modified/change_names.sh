#!/bin/bash

# Recorrer todos los archivos .txt en el directorio actual
for archivo in *.txt; do
    # Comprobar si el archivo sigue el formato "captcha_num1_num2_num3.txt"
    if [[ "$archivo" == captcha_* ]]; then
        # Extraer el nuevo nombre eliminando el prefijo "captcha_"
        nuevo_nombre="${archivo#captcha_}"
        
        # Renombrar el archivo
        mv "$archivo" "$nuevo_nombre"
        
        # Mostrar mensaje de renombrado
        echo "Renombrado: $archivo -> $nuevo_nombre"
    fi
done
