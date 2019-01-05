######################################################################################################################################################################
#                                                                                                                                                                    #
# - Universidad Nacional Autonoma de Mexico                                                                                                                          #
# - Centro de Fisica Aplicada y Tecnologia Avanzada                                                                                                                  #
# - Licenciatura en Tecnologia                                                                                                                                       #
# - Tesista: Marcos Emmanuel Gonzalez Laffitte                                                                                                                       #
# - Tutor: Dra. Maribel Hernandez Rosales                                                                                                                            #
# - Instituto: Instituto de Matematicas UNAM                                                                                                                         #
# - Trabajo: Feed Forwar Artificial Neural Network - 1 - Launcher                                                                                                    #
# - Descripción: Recibe el archivo de entrenamiento y manda llamar a builder, trainer y trainer                                                                      #
# - Instrucciones: los archivos launcher, builder, trainer y tester debe estar en la misma carpeta junto con los archivos de functionTrain.txt y functionTest.txt    #
#                  correr en terminal linux como:                                                                                                                    #
#                        python3.5 FFANN_1_Launcher_Marcos_Emmannuel_Gonzalez_Laffitte.py [function_entrenamiento.txt] [function_prueba.txt]                         #
#   * el archivo debe contener el nombre de la funcion y los formatos señalados junto con su proposito, i.e. _entrenamiento.txt y _prueba.txt                        #
#                                                                                                                                                                    #
#     ejemplo:     python3.5   FFANN_1_Launcher_Marcos_Emmannuel_Gonzalez_Laffitte.py   XOR_entrenamiento.txt   XOR_prueba.txt                                       #
#                                                                                                                                                                    #
#   * se recomienda entrenar a la red con aproximadamente 50 pares de entrenamiento.                                                                                 #
#                                                                                                                                                                    #
# - Fecha: 08 de mayo de 2018                                                                                                                                        #
# - Version del programa: 1.0                                                                                                                                        #
# - Lenguaje: Python                                                                                                                                                 #
# - Version de Lenguaje: Python 3.5.3 :: Anaconda custom (64-bit)                                                                                                    #
#                                                                                                                                                                    #
# ***** SI LOS COMENTARIOS NO SE AJUSTAN A LA PANTALLA, MODIFICAR RESOLUCION DE IDE                                                                                  #
#                                                                                                                                                                    #
######################################################################################################################################################################
# Código #############################################################################################################################################################

# Informacion sobre las dependencias #################################################################################################################################
"""
- sys: integrada en version de python
"""

# Dependencias #######################################################################################################################################################
# dependencias de terceros
import sys                        # obtener entrada en linea de comandos y salir del systema en caso de error

# dependencias propias
import FFANN_2_Builder_Marcos_Emmannuel_Gonzalez_Laffitte as builder
import FFANN_3_Trainer_Marcos_Emmannuel_Gonzalez_Laffitte as trainer
import FFANN_4_Tester_Marcos_Emmannuel_Gonzalez_Laffitte as tester

# Variables globales #################################################################################################################################################
# nombre de funcion a aprender: funcion_entrenamiento.txt & funcion_prueba.txt -> funcion_FFANNmodel.pickle & funcion_FFANNtrained.pickle 
funcion = ""
# conjunto de entrenamiento
setEntrenam = dict()
# conjunto de prueba
setDePrueba = dict()
# factor de normalizacion
factorNorm = 0

# Funciones ##########################################################################################################################################################
# funcion: revisar nombre --------------------------------------------------------------------------------------------------------------------------------------------
def revisarEntrada():
    # variables locales
    archivoEntrenArreglo = []
    archivoPruebaArreglo = []
    archivoEntrenArreglo = sys.argv[1].split("_")
    archivoPruebaArreglo = sys.argv[2].split("_")
    # revisar error: longitud de entrada incorrecta
    if(not len(sys.argv) == 3):
        sys.exit("\n\t\tFFANNexception: El comando proporcionado es incorrecto.\n\n")
    # revistar formato de archivo .txt hasta el final en archivos
    if((not archivoEntrenArreglo.pop() == "entrenamiento.txt") or (not archivoPruebaArreglo.pop() == "prueba.txt")):
        sys.exit("\n\t\tFFANNexception: Alguno de los archivos no tiene formato valido.\n\n")
    # revisar que el nombre de la funcion sea igual en ambos casos
    if(not "_".join(archivoEntrenArreglo) == "_".join(archivoPruebaArreglo)):
        sys.exit("\n\t\tFFANNexception: Los archivos de entrenamiento y prueba no corresponden a la misma funcion.\n\n")
    # revisar que el archivo se encuentere en el mismo directorio
    if(("/" in sys.argv[1]) or ("\\" in sys.argv[1]) or ("/" in sys.argv[2]) or ("\\" in sys.argv[2])):
        sys.exit("\n\t\tFFANNexception: Alguno de los archivos no se encuentra en el directorio actual.\n\n")
    # fin de la funcion
        
# funcion: obtener nombre de funcion a aprender ----------------------------------------------------------------------------------------------------------------------
def obtenerNombreFuncion():
    # mensaje de funcion
    print("\t\t- Obteniendo nombre de la funcion a aprender ...")
    # variables locales
    archivo1 = ""
    archivo1Arreglo = []
    nombre = ""
    # obtener nombr de funcion dese argv
    archivo1 = sys.argv[1]
    archivo1Arreglo = archivo1.split("_")
    archivo1Arreglo.pop()
    nombre = "_".join(archivo1Arreglo)
    # fin de funcion
    return(nombre)
    
# funcion: revisar y obtener archivos de entrenamiento y prueba ------------------------------------------------------------------------------------------------------
def obtenerArchivo(nombre, tipo, formato, escala):
    # mensaje de funcion
    print("\t\t- Analizando y Extrayendo conjunto de " + tipo + " ...")
    # variables locales
    linea = None
    lineaArreglo = []
    xArreglo = []
    yArreglo = []
    xtotal = 0
    ytotal = 0
    contador = 0
    conjunto = dict()
    elemento = None
    maximoAux = 0
    maximo = 0
    # abrir archivo
    archivo = open(nombre + tipo + formato, "r")
    # extraer archivo
    for linea in archivo:
        # partir linea
        if(">" in linea):
            lineaArreglo = linea.strip().split(">")
        else:
            sys.exit("\n\t\tFFANNexception: Se encontro una linea invalida en archivo de " + tipo + ".\n\n")
        # partir entradas y salidas
        if((len(lineaArreglo) != 2) or (lineaArreglo[0] == "") or (lineaArreglo[1] == "")):
            sys.exit("\n\t\tFFANNexception: Se encontro una linea invalida en archivo de " + tipo + ".\n\n")
        else:
            xArreglo = [float(elemento) for elemento in lineaArreglo[0].split(",")]
            yArreglo = [float(elemento) for elemento in lineaArreglo[1].split(",")]
        # revisar longitud de arreglos
        if((xtotal == 0) and (ytotal == 0)):
            xtotal = len(xArreglo)
            ytotal = len(yArreglo)
        if((xtotal != len(xArreglo)) or (ytotal != len(yArreglo))):
            sys.exit("\n\t\tFFANNexception: Se encontro una linea invalida en archivo de " + tipo + ".\n\n")
        # guardar entradas y salidas en hash
        conjunto[str(contador) + "e"] = xArreglo
        conjunto[str(contador) + "s"] = yArreglo
        # obtener maximo de los arreglo
        maximoAux = abs(max(xArreglo))
        if(maximo < maximoAux):
            maximo = maximoAux
        # incrementar contador
        contador = contador + 1
    # revisar error de archivo vacio
    if(contador == 0):
        sys.exit("\n\t\tFFANNexception: El archivo de " + tipo + " se encuentra vacio.\n\n")
    # cerrar archivo
    archivo.close()
    # fin de la funcion
    return(conjunto, maximo)
        
# funcion: comparar archivo de entrenamiento y archivo de prueba -----------------------------------------------------------------------------------------------------
def evaluarConsistenciaArchivos(entrenamiento, prueba):
    # mensaje de funcion
    print("\t\t- Verificando consistencia entre archivos de entrenamiento y prueba ...")
    # verificar que las entradas tengan la misma longitud
    if((len(entrenamiento["0e"])) != (len(prueba["0e"]))):
        sys.exit("\n\t\tFFANNexception: La cantidad de entradas en los archivos de entrenamiento y prueba es inconsistente.\n\n")
    # verificar que las salidas tengan la misma longitud
    if((len(entrenamiento["0s"])) != (len(prueba["0s"]))):
        sys.exit("\n\t\tFFANNexception: La cantidad de salidas en los archivos de entrenamiento y prueba es inconsistente.\n\n")
    # fin de funcion

# funcion: comparar archivo de entrenamiento y archivo de prueba -----------------------------------------------------------------------------------------------------
def normalizarDatos(entrenamiento, prueba, escala):
    # mensaje
    print("\t\t- Verificando factor de escalamiento ...")
    # variables locales
    par = 0
    valor = 0
    llave = ""
    # revisar que factor de escalamiento sea diferente de cero
    if(escala == 0):
        sys.exit("\n\t\tFFANNexception: Todos los datos son cero.\n\n")
    # mensaje
    print("\t\t- Normalizando datos ...")
    # normalizar datos de entrenamiento
    for par in range(int(len(entrenamiento.keys())/2)):
        llave = str(par) + "e"
        for valor in range(len(entrenamiento[llave])):
            entrenamiento[llave][valor] = (entrenamiento[llave][valor]) / (escala)
        llave = str(par) + "s"
        for valor in range(len(entrenamiento[llave])):
            entrenamiento[llave][valor] = (entrenamiento[llave][valor]) / (escala)
    # normalizar datos de prueba
    for par in range(int(len(prueba.keys())/2)):
        llave = str(par) + "e"
        for valor in range(len(prueba[llave])):
            prueba[llave][valor] = (prueba[llave][valor]) / (escala)
        llave = str(par) + "s"
        for valor in range(len(prueba[llave])):
            prueba[llave][valor] = (prueba[llave][valor]) / (escala)
    # fin de funcion
    
# Main ###############################################################################################################################################################

# Validacion de datos de entrada -------------------------------------------------------------------------------------------------------------------------------------
# mensaje incial
print("\n\n\t\t>  Creador de modelos en Redes Neuronales Aritificiales de Propagacion hacia el Frente.")
# checar que el nombre sea correcto e indicar archivos
revisarEntrada()
print("\t\t* Archivo de entrenamiento:\t" + sys.argv[1])
print("\t\t* Archivo de prueba:\t\t" + sys.argv[2])
# mensaje
print("\n\t\t+ Iniciando validacion de archivos proporcionados.")
# obtener nombre de la funcion sin los formatos solicitados
funcion = obtenerNombreFuncion()
# verificar y obtener archivo de entrenamiento
(setEntrenam, factorNorm) = obtenerArchivo(funcion + "_", "entrenamiento", ".txt", factorNorm)
# verificar y obtener archivo de prueba
(setDePrueba, factorNorm) = obtenerArchivo(funcion + "_", "prueba", ".txt", factorNorm)
# evaluar consistencia de conjuntos de entrenamiento y prueba
evaluarConsistenciaArchivos(setEntrenam, setDePrueba)
# normalizar datos
normalizarDatos(setEntrenam, setDePrueba, factorNorm)
# mensaje
print("\t\t- Validacion completada.")

# Construccion del ffann ---------------------------------------------------------------------------------------------------------------------------------------------
# mensaje
print("\n\t\t+ Iniciando construccion del modelo.")
# construir modelo
builder.construirModelo(funcion, setEntrenam, setDePrueba, factorNorm)
# mensaje
print("\t\t- Construccion completada.")

# Train ffann --------------------------------------------------------------------------------------------------------------------------------------------------------
# mensaje
print("\n\t\t+ Iniciando entrenamiento del modelo.")
# entrenar modelo
trainer.obtenerModeloEntrenado(funcion)
# mensaje
print("\t\t- Entrenamiento completado.")

# Test ffann ---------------------------------------------------------------------------------------------------------------------------------------------------------
# mensaje
print("\n\t\t+ Iniciando evaluacion del modelo.")
# evaluar modelo entrenado
tester.obtenerEvaluacionDelModelo(funcion)
# mensaje
print("\t\t- Evaluacion completada.")

# Mensaje de salida --------------------------------------------------------------------------------------------------------------------------------------------------
print("\n\t\t- Se concluyo satisfactoriamente la creacion del modelo solicitado.\n\n")

# Fin ################################################################################################################################################################
######################################################################################################################################################################
