######################################################################################################################################################################
#                                                                                                                                                                    #
# - Universidad Nacional Autonoma de Mexico                                                                                                                          #
# - Centro de Fisica Aplicada y Tecnologia Avanzada                                                                                                                  #
# - Licenciatura en Tecnologia                                                                                                                                       #
# - Tesista: Marcos Emmanuel Gonzalez Laffitte                                                                                                                       #
# - Tutor: Dra. Maribel Hernandez Rosales                                                                                                                            #
# - Instituto: Instituto de Matematicas UNAM                                                                                                                         #
# - Trabajo: Feed Forwar Artificial Neural Network - 3 - Trainer                                                                                                     #
# - Descripción: entrenar el modelo ffann por medio de los datos del archivo de entrenamiento hasta que se alcance el numero de iteraciones deseadas.                #
#                                                                                                                                                                    #
# - Recibe: pickle con modelo de red. El modelo consiste de una tupla con los siguientes elementos:                                                                  #
#           modelo = (funcion, dict de topologia, dict de valores en neuronas, vecindario de entrada, v. de salida, set para entrenamiento, s. p. evaluacion ...     #
#                        0             1                     2                         3                  4                  5                     6                 #
#                    ... factor de normalizacion)                                                                                                                    #
#                                   7                                                                                                                                #
#                                                                                                                                                                    #
#   * se recomienda entrenar a la red con aproximadamente 50 pares de entrenamiento.                                                                                 #
#                                                                                                                                                                    #
# - Salida: pickle con modelo entrenado y grafica de convergencia de la funcion de error para modelo entrenado.                                                      #
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

- random: integrada en version de python

- matplotlib: 2.0.2; Home page: http://matplotlib.org

- warnings: integrada en version de python

- sys: integrada en version de python
"""

# Dependencias #######################################################################################################################################################
import sys                        # erase line from begining
import pickle                     # guardar objetos en archivo de bits
import math                       # funcion de activacion
import random                     # use randrange to generate random intial weights for the ffann and sgd
import matplotlib.pyplot as plt   # graficar la convergencia del error
import warnings                   # ignorar advertencias de matplotlib en servidor
import time
warnings.simplefilter("ignore")       
plt.switch_backend('agg')

# Funciones ##########################################################################################################################################################
# funcion: calcular funcion de activacion ----------------------------------------------------------------------------------------------------------------------------
def phi(x_v):
    # variables locales
    Phi = 0
    # calcular funcion de activacion
    Phi = (math.exp(x_v)) / (math.exp(x_v) + 1)
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
    PhiD = (phiv)*(1 - phiv)
    # fin de la funcion
    return(PhiD)

# funcion: abrir archivo pickle para obtener modelo de red -----------------------------------------------------------------------------------------------------------
def abrirArchivo(archivo):
    # variables locales
    someFile = None
    red = []
    # abrir archivo de modelo
    someFile = open(archivo + "_FFANNmodel.pickle", "rb")
    # desempaquetar pickle
    red = pickle.load(someFile)
    someFile.close()
    # fin de la funcion
    return(red)

# funcion: iniciar pesos de la red con valores aleatorios entre -cota y cota -----------------------------------------------------------------------------------------
def inicializarPesosAleatorios(vecSalida, cota, topo):
    # variables locales
    prof = 0
    capa = 0
    lugar = 0
    neurona = ""
    vecinos = 0
    pesos = []
    peso = 0
    factorPesos = 100000
    # obtener profundidad de la red
    prof = len(topo) - 1
    # recorrer las capas internas modificando los pesos en los arcos
    for capa in range(prof - 2):
        for lugar in range(topo[capa]):
            pesos = []
            neurona = str(lugar) + "_" + str(capa)
            vecinos = len(vecSalida[neurona][0])
            [pesos.append(random.randrange(-1*factorPesos, 1*factorPesos, 1)*(float(cota)/factorPesos)) for peso in range(vecinos)]
            vecSalida[neurona][1] = pesos
    # fin de la funcion
    
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
        valNeuronas[neurona][1] = (1/2)*((senal) - (valSalida[lugar]))*((senal) - (valSalida[lugar]))
        valNeuronas[neurona][2] = (senal) - (valSalida[lugar])
    # realizar propagacion hacia el frente para la segunda etapa del modulo error
    neurona = "0_" + str(prof)
    senal = 0
    for vecino in vecEntrada[neurona]:
        senal = (senal) + (valNeuronas[vecino][1])
    valNeuronas[neurona][0] = senal
    valNeuronas[neurona][1] = senal
    valNeuronas[neurona][2] = 1
    # fin de la funcion
    
# funcion: retropropagacion para capas restantes ---------------------------------------------------------------------------------------------------------------------
def retropropagacion(valNeuronas, vecSalida, topo):
    # variables locales
    neurona = ""
    lugar = 0
    capa = 0
    prof = 0
    vecino = None
    indice = 0
    d_n = 0
    errProp = 0
    # obtener profundidad
    prof = len(topo) - 1
    # retropropagacion para la segunda etapa del modulo de evaluacion de error
    neurona = "0_" + str(prof)
    valNeuronas[neurona][3] = valNeuronas[neurona][2]
    # retropropagacion para capas restantes
    for capa in reversed(range(prof)):
        for lugar in range(topo[capa]):
            neurona = str(lugar) + "_" + str(capa)
            d_n = 0
            for vecino in vecSalida[neurona][0]:
                indice = vecSalida[neurona][0].index(vecino)
                peso = vecSalida[neurona][1][indice]
                errProp = valNeuronas[vecino][3]
                d_n = (d_n) + ((peso)*(errProp))
            d_n = (d_n)*(valNeuronas[neurona][2])
            valNeuronas[neurona][3] = d_n
    # fin de la funcion

# funcion: actualizacion de pesos ------------------------------------------------------------------------------------------------------------------------------------
def actualizarPesos(valNeuronas, vecSalida, ritmo, masa, topo):
    # variables locales
    senal = 0
    errProp = 0
    prof = 0
    neurona = ""
    vecino = None
    indice = 0
    d_w = 0
    velocidad = 0
    momento = 0
    # obtener profundidad
    prof = len(topo) - 1
    # recorrer la sub red neuronal por capas modificando los pesos (el modulo error no se debe modificar)
    for capa in range(prof - 2):
        for lugar in range(topo[capa]):
           neurona = str(lugar) + "_" + str(capa)
           for vecino in vecSalida[neurona][0]:
               d_w = 0
               senal = valNeuronas[neurona][1]
               errProp = valNeuronas[vecino][3]
               indice = vecSalida[neurona][0].index(vecino)
               velocidad = vecSalida[neurona][2][indice]
               momento = (masa)*(velocidad)
               d_w = (ritmo)*(senal)*(errProp)
               vecSalida[neurona][1][indice] = (vecSalida[neurona][1][indice]) - (d_w) + (momento)
               vecSalida[neurona][2][indice] = d_w
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

# funcion: entrenar modelo -------------------------------------------------------------------------------------------------------------------------------------------
def entrenarModelo(modelo, errorCota, ritmoApr, inercia, intervalo, iteraciones):
    # mensaje
    print("\t\t- Entrenando modelo ...")
    # variables locales
    errorProm = 1
    convError = []
    contador = 0
    it = 0
    ffann = ()
    entrada = []
    salida = []
    funcion = ""
    topologia = []
    neuronas = dict()
    vecEnt = dict()
    vecSal = dict()
    setEnt = dict()
    setEva = dict()
    normal = 0
    tiempoInicial = 0
    tiempoFinal = 0
    tiempo = 0
    recorridas = []
    llavesAleatorias = []
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
    # tomar llaves
    llavesAleatorias = list(setEnt.keys())
    #print(llavesAleatorias)
    # inicializar pesos de la red
    inicializarPesosAleatorios(vecSal, intervalo, topologia)
    # Obtener valor para el error deseado equivalente a una reduccion del 95% del error inicial
    errorProm = 0
    recorridas = []
    random.shuffle(llavesAleatorias)
    for llave in llavesAleatorias:
        if(not llave in recorridas):
            # definir llaves
            llaveE = "".join((list(llave)[0 : len(llave) - 1])) + "e"
            llaveS = "".join((list(llave)[0 : len(llave) - 1])) + "s"
            # definir par de entrenamiento
            entrada = setEnt[llaveE]#setEnt[str(contador) + "e"]
            salida = setEnt[llaveS]#setEnt[str(contador) + "s"]
            # iniciar neuronas en capa de entrada con par de entrenamiento
            activarCapaDeEntrada(neuronas, entrada)
            # propagacion hacia el frente hasta capa de salida
            propagacionHastaCapaSalida(neuronas, vecEnt, vecSal, topologia)
            # propagacion hacia el frente para modulo de evaluacion de error
            propagacionModuloError(neuronas, vecEnt, salida, topologia)
            # obtener error
            errorProm = (errorProm) + (obtenerError(neuronas, topologia))
            # llaves ya pasadas
            recorridas.append(llaveE)
            recorridas.append(llaveS)
    errorProm = (errorProm) / (len(setEnt.keys())/2)    
    errorDeseado = ((errorProm) / (100)) * (100 - errorCota)
    # ciclo de entrenamiento guiado por la reduccion del error hasta alcanzar el error deseado
    errorProm = errorDeseado
    tiempoInicial = time.time()
    #while(errorProm >= errorDeseado):
    while(it < iteraciones):
        # incializar error promedio en ceros
        errorProm = 0
        recorridas = []
        random.shuffle(llavesAleatorias)
        for llave in llavesAleatorias:
            if(not llave in recorridas):
                # definir llaves
                llaveE = "".join((list(llave)[0 : len(llave) - 1])) + "e"
                llaveS = "".join((list(llave)[0 : len(llave) - 1])) + "s"
                #print(llaveE, llaveS)
                # definir par de entrenamiento
                entrada = setEnt[llaveE]#setEnt[str(contador) + "e"]
                salida = setEnt[llaveS]#setEnt[str(contador) + "s"]
                # iniciar neuronas en capa de entrada con par de entrenamiento
                activarCapaDeEntrada(neuronas, entrada)
                # propagacion hacia el frente hasta capa de salida
                propagacionHastaCapaSalida(neuronas, vecEnt, vecSal, topologia)
                # propagacion hacia el frente para modulo de evaluacion de error
                propagacionModuloError(neuronas, vecEnt, salida, topologia)
                # paso de retropropagacion
                retropropagacion(neuronas, vecSal, topologia)
                # actualizar pesos
                actualizarPesos(neuronas, vecSal, ritmoApr, inercia, topologia)
                # obtener error
                errorProm = (errorProm) + (obtenerError(neuronas, topologia))
                # llaves ya pasadas
                recorridas.append(llaveE)
                recorridas.append(llaveS)
        #print("\n")
        # promediar error
        errorProm = (errorProm) / (len(setEnt.keys())/2)
        # agregar error promedio a curva de aprendizaje
        convError.append(errorProm)
        # imprimir error
        print("", end = "\r")
        print("\t\t- Total de iteraciones solicitadas: "
              + str(iteraciones)
              + "\t - avance: % "
              + str(round((it * 100)/iteraciones , 2)),
              end = "\r")
        it = it + 1
    tiempoFinal = time.time()
    # obtener tiempo
    tiempo = round((tiempoFinal - tiempoInicial) / 60, 5)
    # mensaje de aprendizaje finalizado
    sys.stdout.write("\033[K")
    print("\t\t- Finalizo el entrenamiento con error de: " + str(round(errorProm, 5)) + "\t- Iteraciones: " + str(it) + " ...")     
    # fin de la funcion
    return(ffann, errorProm, convError, tiempo)

# funcion: imprimir curva de aprendizaje -----------------------------------------------------------------------------------------------------------------------------
def imprimirCurvaAprendizaje(nombreSucio, error, errorFinal, tiempo):
    # mensaje
    print("\t\t- Imprimiendo curva de aprendizaje ...")
    # variables locales
    nombreArreglo = []
    nombre = ""
    titulo = ""
    curvaStr = ""
    funciStr = ""
    errorStr = ""
    iteraStr = ""
    # obtener nombre de la funcion
    nombreArreglo = nombreSucio.split("_")
    nombre = " ".join(nombreArreglo)
    # definir titulo
    curvaStr = "Curva de Aprendizaje de la FFANN"
    funciStr = "Funcion aprendida: "
    tiempoStr = "Tiempo: "
    errorStr = "Error mínimo alcanzado: "
    iterStr = "Numero de Iteraciones : "
    titulo =  curvaStr + "\n"  + funciStr + nombre  + "\n" + tiempoStr + str(tiempo) + "\n" + errorStr + str(round(errorFinal, 5)) + " - " + iterStr + str(len(error))
    # imprimir grafica de curva de aprendizaje
    plt.plot([errorFinal]*len(error), linestyle = "--", color = "lightgray")
    plt.plot(error, linestyle = '-', color = 'midnightblue')
    plt.locator_params(nbins = 6)
    plt.xlim(0, len(error))
    plt.title(titulo, fontsize = 10)
    plt.xlabel("Iteraciones sobre el conjunto de entrenamiento", fontsize = 10)
    plt.ylabel("Error promedio", fontsize = 10)
    plt.tight_layout()
    plt.savefig("Curva_de_Aprendizaje_" + nombreSucio + ".pdf")
    # fin de la funcion

# funcion: guardar modelo entrenado en archivo pickle ----------------------------------------------------------------------------------------------------------------
def guardarModeloEntrenado(red):
    # mensaje
    print("\t\t- Guardando modelo entrenado ...")
    # variables locales
    archivo = None
    # guardar modelo
    archivo = open(red[0] + "_FFANNtrained.pickle", "wb")
    pickle.dump(red, archivo)
    archivo.close()    
    # fin de la funcion

# Main ###############################################################################################################################################################
def obtenerModeloEntrenado(funcion):
    # parametros de aprendizaje
    errorParaReducir = 99
    ritmoAprendizaje = 1
    masaInercial = 0
    cotaIntervalo = 1
    iteraciones = 10000
    tiempoTot = 0
    # variables locales
    modeloEntrenado = ()
    errorPromFinal = 0
    convError = []
    # entrenar modelo
    (modeloEntrenado, errorPromFinal, convError, tiempoTot) = entrenarModelo(funcion, errorParaReducir, ritmoAprendizaje, masaInercial, cotaIntervalo, iteraciones)
    # generar grafica de curva de aprendizaje
    imprimirCurvaAprendizaje(funcion, convError, errorPromFinal, tiempoTot)
    # guardar modelo entrenado
    guardarModeloEntrenado(modeloEntrenado)
    # fin de la funcion
    
# Fin ################################################################################################################################################################
######################################################################################################################################################################
