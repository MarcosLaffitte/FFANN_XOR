######################################################################################################################################################################
#                                                                                                                                                                    #
# - Universidad Nacional Autonoma de Mexico                                                                                                                          #
# - Centro de Fisica Aplicada y Tecnologia Avanzada                                                                                                                  #
# - Licenciatura en Tecnologia                                                                                                                                       #
# - Tesista: Marcos Emmanuel Gonzalez Laffitte                                                                                                                       #
# - Tutor: Dra. Maribel Hernandez Rosales                                                                                                                            #
# - Instituto: Instituto de Matematicas UNAM                                                                                                                         #
# - Trabajo: Feed Forwar Artificial Neural Network - 2 - Builder                                                                                                     #
# - Descripción: Construir una FFANN , dada una topologia de propagacion hacia el frente en su descripcion general ...                                               #
#                                                                                                                                                                    #
# - Recibe:  Archivo de entrenamiento con formato genreral (funcion_a_aprender.txt, sin espacios)...                                                                 #
#            1. x11, x12, ..., x1a > y11, y12, ..., y1b                                                                                                              #
#            2. x21, x22, ..., x2a > y21, y22, ..., y2b                                                                                                              #
#            .             .              .           .                                                                                                              #
#            .             .              .           .                                                                                                              #
#            .             .              .           .                                                                                                              #
#            m. xm1, xm2, ..., xma > ym1, ym2, ..., ymb                                                                                                              #
#                                                                                                                                                                    #
#            - topologia = [a, h, b, b , 1] , donde                                                                                                                  #
#            - cada entrada de la lista topologia es una capa e indica la cantidad de neuronas en esa capa                                                           #
#            - a = no. de neuronas en capa de entrada                                                                                                                #
#            - b = no. de neuronas en la capa de salida                                                                                                              #
#            - h = 5                                                                                                                                                 #
#            - m = cantidad de pares de entrenamiento                                                                                                                #
#            - ultimas dos capas de topologia = modulo de evaluacion del error                                                                                       #
#                                                                                                                                                                    #
#   * se recomienda entrenar a la red con aproximadamente 50 pares de entrenamiento.                                                                                 #
#                                                                                                                                                                    #
# - Salida: Red neuronal recurrente en un pickle de la tupla (nombre, topologia, tuplas, vecEntrada, vecSalida, setParaEntren, setParaPrueba, escala)                #
#                                                                                                                                                                    #
# - Fecha: 10 de mayo de 2018                                                                                                                                        #
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
- pickle: integrada en version de python

- sys: integrada en version de python
"""

# Dependencias #######################################################################################################################################################
import pickle                     # guardar objetos en archivo de bit

# Funciones ##########################################################################################################################################################
# funcion: construri arreglo de topologia ----------------------------------------------------------------------------------------------------------------------------
def construirTopologia(conjunto):
    # mensaje de funcion
    print("\t\t- Construyendo arreglo de topologia ...")
    # variables locales
    cardCapaEntrada = 0
    cardCapaSalida = 0
    numCapasInternas = 0
    cardCapaInterna = 0
    contador = 0
    topo = []
    # obtener datos sobre el conjunto de entrenamiento
    cardCapaEntrada = len(conjunto["0e"])
    cardCapaSalida = len(conjunto["0s"])
    cardCapaInterna = 5
    # construccion de arreglo de topologia
    topo.append(cardCapaEntrada)
    topo.append(cardCapaInterna)
    topo.append(cardCapaSalida)
    topo.append(cardCapaSalida)
    topo.append(1)
    # imprimir arreglo de topologia
    print("\t\t- topologia: ", end = "")
    print(topo)
    # fin de la funcion
    return(topo)

# funcion: incializar hash tupples -----------------------------------------------------------------------------------------------------------------------------------
def inicializarTuplas(topo):
    # mensaje de funcion
    print("\t\t- Inicializando tabla hash de tuplas ...")
    # variables locales
    profundidad = 0
    valores = dict()
    capa = 0
    lugar = 0
    neurona = ""
    # obtener profundidad de la red
    profundidad = len(topo)
    # crear nodos con sus respectivas entradas
    for capa in range(profundidad):
        for lugar in range(topo[capa]):
            neurona = str(lugar) + "_" + str(capa) 
            valores[neurona] = [0,0,0,0]    
    # fin de la funcion
    return(valores)

# funcion: incializar hash inHood -----------------------------------------------------------------------------------------------------------------------------------
def inicializarVecindarioEntrada(topo):
    # mensaje de funcion
    print("\t\t- Inicializando tabla hash de vecindario de entrada ...")
    # variables locales
    profundidad = 0
    vecinos = []
    condominio = dict()
    capa = 0
    lugar = 0
    neurona = ""
    contador = 0
    # obtener profundidad
    profundidad = len(topo) - 1
    # definir vecindarios de entrada para capa de salida y capas ocultas o intermedias
    for capa in range(1, profundidad - 1):
        for lugar in range(topo[capa]):
            neurona = str(lugar) + "_" + str(capa)
            vecinos = []
            [vecinos.append(str(contador) + "_" + str(capa - 1)) for contador in range(topo[capa - 1])]
            condominio[neurona] = vecinos
    # definir vecindarios de entrada para nodos en la etapa 1 de modulo para evaluacion de error
    for lugar in range(topo[profundidad - 1]):
        condominio[str(lugar) + "_" + str(profundidad - 1)] = [str(lugar) + "_" + str(profundidad - 2)]
    # definir vecindarios de entrada para nodos en la etapa 2 de modulo para evaluacion de error
    vecinos = []
    [vecinos.append(str(contador) + "_" + str(profundidad - 1)) for contador in range(topo[profundidad - 1])]
    condominio["0_" + str(profundidad)] = vecinos
    # fin de la funcion
    return(condominio)
    
# funcion: incializar hash outHood -----------------------------------------------------------------------------------------------------------------------------------
def inicializarVecindarioSalida(topo):
    # mensaje de funcion
    print("\t\t- Inicializando tabla hash de vecindario de salida ...")
    # variables locales
    profundidad = 0
    vecinos = []
    pesos = []
    velocidades = []
    vecindario = []
    condominio = dict()
    capa = 0
    lugar = 0
    neurona = ""
    contador = 0
    # obtener profundidad
    profundidad = len(topo) - 1
    # definir vecindarios de salida para capa de entrada y capas ocultas o intermedias
    for capa in range(profundidad - 2):
        for lugar in range(topo[capa]):
            vecindario = []
            vecinos = []
            pesos = []
            velocidades = []
            [vecinos.append(str(contador) + "_" + str(capa + 1)) for contador in range(topo[capa + 1])]
            pesos = [0]*len(vecinos)
            velocidades = [0]*len(vecinos)
            vecindario = [vecinos, pesos, velocidades]
            neurona = str(lugar) + "_" + str(capa)
            condominio[neurona] = vecindario
    # definir vecindarios de salida para nodos en la capa de salida
    capa = profundidad - 2
    for lugar in range(topo[capa]):
        condominio[str(lugar) + "_" + str(capa)] = [[str(lugar) + "_" + str(capa + 1)], [1], [0]]
    # definir vecindarios de salida para nodos en etapa 1 de modulo para evaluacion de error
    capa = profundidad - 1
    for lugar in range(topo[capa]):
        condominio[str(lugar) + "_" + str(capa)] = [["0_" + str(capa + 1)], [1], [0]]
    # fin de la funcion
    return(condominio)

# funcion: construir red y guardar en pickle -------------------------------------------------------------------------------------------------------------------------
def GuardarRed(red):
    # mensaje
    print("\t\t- Guardando modelo en archivo pickle ...")
    # variables locales
    archivo = None
    # abrir archivo
    archivo = open(red[0] + "_FFANNmodel.pickle", "wb")
    # guardar
    pickle.dump(red, archivo)
    # cerrar archivo
    archivo.close()
    # fin de la funcion
    
# Main ###############################################################################################################################################################

def construirModelo(nombre, setParaEntren, setParaPrueba, escala):
    # variables locales
    topologia = []
    tuplas = dict()
    vecEntrada = dict()
    vecSalida = dict()
    ffann = ()
    # construir topologia
    topologia = construirTopologia(setParaEntren)
    # construir tuplas
    tuplas = inicializarTuplas(topologia)
    # construir lista de vecindarios de entrada
    vecEntrada = inicializarVecindarioEntrada(topologia)    
    # construir lista de vecindarios de salida
    vecSalida = inicializarVecindarioSalida(topologia)
    # construir red
    ffann = (nombre, topologia, tuplas, vecEntrada, vecSalida, setParaEntren, setParaPrueba, escala)
    GuardarRed(ffann)
    # fin de la funcion
    
# Fin ################################################################################################################################################################
######################################################################################################################################################################
