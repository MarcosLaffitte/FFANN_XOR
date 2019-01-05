######################################################################################################################################################################
#                                                                                                                                                                    #
# - Universidad Nacional Autonoma de Mexico                                                                                                                          #
# - Centro de Fisica Aplicada y Tecnologia Avanzada                                                                                                                  #
# - Licenciatura en Tecnologia                                                                                                                                       #
# - Tesista: Marcos Emmanuel Gonzalez Laffitte                                                                                                                       #
# - Tutor: Dra. Maribel Hernandez Rosales                                                                                                                            #
# - Instituto: Instituto de Matematicas UNAM                                                                                                                         #
# - Trabajo: Feed Forwar Artificial Neural Network - 4 - Tester                                                                                                      #
# - Descripción: evaluar el modelo ffann por medio de los datos del archivo de prueba.                                                                               #
#                                                                                                                                                                    #
# - Recibe: pickle con modelo de red. El modelo consiste de una tupla con los siguientes elementos:                                                                  #
#           modelo = (funcion, dict de topologia, dict de valores en neuronas, vecindario de entrada, v. de salida, set para entrenamiento, s. p. evaluacion ...     #
#                        0             1                     2                         3                  4                  5                     6                 #
#                    ... factor de normalizacion)                                                                                                                    #
#                                   7                                                                                                                                #
#                                                                                                                                                                    #
#   * se recomienda entrenar a la red con aproximadamente 50 pares de entrenamiento.                                                                                 #
#                                                                                                                                                                    #
# - Salida: archivo de prueba con las entradas dadas junto sus salidas esperadas y reales.                                                                           #
#                                                                                                                                                                    #
# - Fecha: 15 de mayo de 2018                                                                                                                                        #
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

- math: integrada en version de python

- mpMath:      
               - version: 0.19
               - site: http://www.mpmath.org/
               - last checked: August/30/2018
"""

# Dependencias #######################################################################################################################################################
import pickle                     # guardar objetos en archivo de bits
import math                       # funcion de activacion

# Funciones ##########################################################################################################################################################
# funcion: calcular funcion de activacion ----------------------------------------------------------------------------------------------------------------------------
def phi(x_v):
    # variables locales
    Phi = 0
    # calcular funcion de activacion
    Phi = math.exp(x_v) / (math.exp(x_v) + 1)
    # fin de la funcion
    return(Phi)

# funcion: calcular funcion de activacion ----------------------------------------------------------------------------------------------------------------------------
def phiD(x_val):
    # variables locales
    PhiD = 0
    phiv = 0
    # obtener funcion de activacion
    phiv = phi(x_val)
    # calcular derivada de la funcion de activacion
    PhiD = phiv*(1 - phiv)
    # fin de la funcion
    return(PhiD)

# funcion: abrir archivo pickle para obtener modelo de red -----------------------------------------------------------------------------------------------------------
def abrirArchivo(archivo):
    # variables locales
    someFile = None
    red = []
    # abrir archivo de modelo
    someFile = open(archivo + "_FFANNtrained.pickle", "rb")
    # desempaquetar pickle
    red = pickle.load(someFile)
    someFile.close()
    # fin de la funcion
    return(red)
    
# funcion: dar valores de entrenamiento a red ------------------------------------------------------------------------------------------------------------------------
def activarCapaDeEntrada(valNeuronas, valEntrada):
    # variables locales
    lugar = 0
    neurona = ""
    # ciclo para ingresar valor a la neurona de entrada
    for lugar in range(len(valEntrada)):
        neurona = str(lugar) + "_0"
        valNeuronas[neurona][0] = valEntrada[lugar]
        valNeuronas[neurona][1] = phi(valNeuronas[neurona][0])
        valNeuronas[neurona][2] = phiD(valNeuronas[neurona][0])
    # fin de la funcion

# funcion: propagacion hacia el frente capas ocultas -----------------------------------------------------------------------------------------------------------------
def propagacionHastaCapaSalida(valNeuronas, vecEntrada, vecSalida, topo):
    # variables locales
    capa = 0
    lugar = 0
    neurona = ""
    x_n = 0
    vecino = None
    indice = 0
    peso = 0
    senal = 0
    prof = 0
    # obtener profundidad
    prof = len(topo) - 1
    # realizar propagacion hacia el frente hasta capa de salida
    for capa in range(1, prof - 1):
        for lugar in range(topo[capa]):
            neurona = str(lugar) + "_" + str(capa)
            x_n = 0
            for vecino in vecEntrada[neurona]:
                indice = vecSalida[vecino][0].index(neurona)
                peso = vecSalida[vecino][1][indice]
                senal = valNeuronas[vecino][1]
                x_n = x_n + (senal*peso)
            valNeuronas[neurona][0] = x_n
            valNeuronas[neurona][1] = phi(x_n)
            valNeuronas[neurona][2] = phiD(x_n)
    # fin de la funcion

# funcion: propagacion hacia el frente modulo de evaluacion de error -------------------------------------------------------------------------------------------------
def propagacionModuloError(valNeuronas, vecEntrada, valSalida, topo):
    # variables locales
    capa = 0
    lugar = 0
    neurona = ""
    vecino = None
    indice = 0
    peso = 0
    senal = 0
    prof = 0
    # obtener profundidad
    prof = len(topo) - 1
    # realizar propagacion hacia el frente para la primera etapa del modulo error
    capa = prof - 1
    for lugar in range(len(valSalida)):
        neurona = str(lugar) + "_" + str(capa)
        vecino = str(lugar) + "_" + str(capa - 1)
        senal = valNeuronas[vecino][1]
        valNeuronas[neurona][0] = senal
        valNeuronas[neurona][1] = (1/2)*(senal - valSalida[lugar])*(senal - valSalida[lugar])
        valNeuronas[neurona][2] = (senal - valSalida[lugar])
    # realizar propagacion hacia el frente para la segunda etapa del modulo error
    neurona = "0_" + str(prof)
    senal = 0
    for vecino in vecEntrada[neurona]:
        senal = senal + valNeuronas[vecino][1]
    valNeuronas[neurona][0] = senal
    valNeuronas[neurona][1] = senal
    valNeuronas[neurona][2] = 1
    # fin de la funcion    

# funcion: obtencion de error promedio -------------------------------------------------------------------------------------------------------------------------------
def obtenerError(valNeuronas, topo):
    # variables locales
    error = 0
    prof = 0
    neurona = ""
    # obtener profundidad
    prof = len(topo) - 1
    # obtener error
    neurona = "0_" + str(prof)
    error = valNeuronas[neurona][1]
    # fin de la funcion
    return(error)

# funcion: obtencion de error promedio -------------------------------------------------------------------------------------------------------------------------------
def obtenerPrediccion(valNeuronas, topo):
    # variables locales
    capa = 0
    lugar = 0
    prof = 0
    neurona = ""
    prediccionSalida = []
    # obtener profundidad
    prof = len(topo) - 1
    # obtener salida
    capa = prof - 2
    for lugar in range(topo[capa]):
        neurona = str(lugar) + "_" + str(capa)
        prediccionSalida.append(valNeuronas[neurona][1])
     # fin de la funcion
    return(prediccionSalida)
    
# funcion: entrenar modelo -------------------------------------------------------------------------------------------------------------------------------------------
def evaluarModelo(modelo):
    # mensaje
    print("\t\t- Envaluando aprendizaje ...")
    # variables locales
    errorProm = 0
    contador = 0
    entrada = []
    salida = []
    ffann = ()
    funcion = ""
    topologia = []
    neuronas = dict()
    vecEnt = dict()
    vecSal = dict()
    setEnt = dict()
    setEva = dict()
    normal = 0
    salidaPred = []
    prediccion = dict()
    # abrir archivo de modelo
    ffann = abrirArchivo(modelo)
    # obtener elementos del ffann
    # [0-funcion, 1-topologia, 2-valores, 3-vec.entrada, 4-vec.salida, 5-set de entrenamiento, 6-set de evaluacion]
    funcion = ffann[0]
    topologia = ffann[1]
    neuronas = ffann[2]
    vecEnt = ffann[3]
    vecSal = ffann[4]
    setEnt = ffann[5]
    setEva = ffann[6]
    normal = ffann[7]
    # incializar error promedio en ceros
    errorProm = 0
    # pasos del algoritmo de retropropagacion para cada pareja de entrenamiento
    for contador in range(int(len(setEva.keys())/2)):
        # definir par de entrenamiento
        entrada = setEva[str(contador) + "e"]
        salida = setEva[str(contador) + "s"]
        # iniciar neuronas en capa de entrada con par de entrenamiento
        activarCapaDeEntrada(neuronas, entrada)
        # propagacion hacia el frente hasta capa de salida
        propagacionHastaCapaSalida(neuronas, vecEnt, vecSal, topologia)
        # propagacion hacia el frente para modulo de evaluacion de error
        propagacionModuloError(neuronas, vecEnt, salida, topologia)
        # obtener salida predicha por el modelo
        salidaPred = obtenerPrediccion(neuronas, topologia)
        # conformar salida
        prediccion[str(contador) + "p"] = [setEva[str(contador) + "e"], setEva[str(contador) + "s"], salidaPred]
        # obtener error
        errorProm = errorProm + obtenerError(neuronas, topologia)
    # promediar error
    errorProm = errorProm / (len(setEnt.keys())/2)
    # mensaje
    print("\t\t- Discrepancia con los resultados esperados con error de: " + str(round(errorProm, 7)))
    # fin de la funcion
    return(errorProm, prediccion, normal)

# funcion: imprimir curva de aprendizaje -----------------------------------------------------------------------------------------------------------------------------
def desnormalizarDatos(datos, normal):
    # mensaje
    print("\t\t- Escalando datos a valores originales ....")
    # variables locales
    prediccion = 0
    valor = 0
    llave = ""
    # desnormalizar datos
    for prediccion in range(len(datos.keys())):
        llave = str(prediccion) + "p"
        for valor in range(len(datos[llave][0])):
            datos[llave][0][valor] = datos[llave][0][valor] * normal
        for valor in range(len(datos[llave][1])):
            datos[llave][1][valor] = datos[llave][1][valor] * normal
        for valor in range(len(datos[llave][2])):
            datos[llave][2][valor] = datos[llave][2][valor] * normal
    # fin de la funcion

# funcion: imprimir curva de aprendizaje -----------------------------------------------------------------------------------------------------------------------------
def imprimirArchivoPrediccion(nombre, error, salida):
    # mensaje
    print("\t\t- Imprimiendo archivo con error de prediccion y resultados ...")
    # variables locales
    prediccion = 0
    valor = 0
    llave = ""
    archivoPred = None
    # abrir archivo
    archivoPred = open(nombre + "_prediccion.txt", "w")
    # imprimir error
    archivoPred.write("# error: " + str(error) + "\n")
    # guardar valores
    for prediccion in range(len(salida.keys())):
        llave = str(prediccion) + "p"
        archivoPred.write(",".join(map(str,salida[llave][0])) + ">" + ",".join(map(str,salida[llave][1])) + "|" + ",".join(map(str,salida[llave][2])) + "\n")
    # cerrar archivo
    archivoPred.close()
    # fin de la funcion

# Main ###############################################################################################################################################################
def obtenerEvaluacionDelModelo(funcion):
    # variables locales
    errorPrediccion = 0
    factorNorm = 0
    resultados = dict()
    # evaluar el modelo
    (errorPrediccion, resultados, factorNorm) = evaluarModelo(funcion)
    # multiplicar datos por factor de normalizacion
    desnormalizarDatos(resultados, factorNorm)
    # imprimir archivo de resultados
    imprimirArchivoPrediccion(funcion, errorPrediccion, resultados)
    # fin de la funcion
    
# Fin ################################################################################################################################################################
######################################################################################################################################################################



