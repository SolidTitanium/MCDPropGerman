# Libreta en MD para la primera parte del trabajo escrito de la clase de matemáticas.

## Objetivos:
- Parte 1
    - Definición y ejemplos de redes neuronales
    - Esquema general de las matemáticas en las redes neuronales (con su respectivo diagrama de flujo)
    - Elegir problema de interés
- Parte 2
    - Desarrollo matemático detallado del problema elegido

# ¿Qué son las redes neuronales?

Una red neuronal artificial (o sólo red neuronal) es un sistema computacional que consiste en una colección de nodos a los que llamamos neuronas las cuales están conectadas entre sí, usualmente se encuentran agrupados en capas con las conexiones ocurriendo entre las neuronas de una capa y otra, las neuronas transmiten "señales", las cuales son un número real, entre sus conexiones ajustando la señal de entrada utilizando alguna función no lineal para despues entregar el valor resultando como salida.

## ¿Para qué sirven las redes neuronales?

Debido a su capacidad para reproducir y modelar procesos no lineales de muy alta complejidad, se han encontrado aplicaciones para las redes neuronales en una variedad muy alta de disciplinas. Las áreas de uso incluyen: 

- Procesos de control: administración de recursos, control de sistemas y predicción de trayectorias

- Reconocimiento de patrones: clasificación de señales, reconocimiento facial y sistemas de radar

- Reconocimiento de secuencias: interpretación de textos manuscritos, lenguaje hablado, gestos y expresiones

- Generación de texto, imágenes y video

- Análisis médico

- Economía y finanzas

Entre muchas otras disciplinas y áreas de trabajo donde esta clase de sistemas siguen encontrando aplicaciones.

A continuación se presentan un par de ejemplos concretos:

AlphaZero, CNN (red neuronal convolucional)

![AlphaZeroLogo](https://images.chesscomfiles.com/uploads/v1/chess_term/6679f740-122f-11eb-9cdd-792fd15c63bd.a5fdbee8.5000x5000o.13f0dadcefd9.png)

Este software desarrollado por Google se centra aprender a jugar partidas de juegos generales, principalmente ajedrez, go y shogi, a un nivel donde es capaz de ganar con facilidad incluso a otros sistemas de aprendizaje automático y redes neuronales diseñadas para destacar específicamente en esos juegos. Consiste en una combinación entre una red neuronal de tipo convolucional y un algoritmo de búsqueda de árbol de Monte Carlo, esta combinación le permite aprender de manera muy rápida jugando partidas contra si mismo y alcanzar un nivel profesional en un periodo muy corto de tiempo.

Dall-E 2, Transformer

![Dall-ELogo](https://promptmuse.com/wp-content/uploads/2022/12/OpenAI-Dall-E-2.jpeg)

Desarrollado por OpenAI, Dall-E 2 es un "transformer" (más específicamente un transformer generativo preentrenado o GPT), el cual es un modelo de aprendizaje profundo que se especializa en procesar secuencias de datos tales como lenguaje natural o imágenes. Dall-E 2 utiliza además un modelo de difusión con el que es capaz de generar imágenes en diversos estilos a partir de texto.

## ¿Como se desarrollaron?

Los primeros modelos desarrollados en el área tenian que ver más con métodos numéricos de regresión lineal que con redes neuronales, poco a poco se fueron creando modelos como el *modelo de Ising* (1925) por Wilhelm Lenz y Ernst Ising el cual es esencialmente una red neuronal recursiva sin aprendizaje.

Sin embargo los inicios de las redes neuronales como las conocemos se dieron en 1943 con la publicación de un artículo por el neurofisiologo Warren McCulloch and el matemático Walter Pitts [[1](https://www.cse.chalmers.se/~coquand/AUTOMATA/mcp.pdf)] donde se describia como las neuronas podrian ser modeladas en circuitos eléctricos, despues, en 1949, Donald Hebb escribio "The Organization of Behavior" [[2](https://pure.mpg.de/rest/items/item_2346268_3/component/file_2346267/content)] en donde se ilustra como las conexiones neuronales se fortalezen con cada uso que se les da, un concepto esencial para que se de el fenómeno del aprendizaje, esta serie de avances dio lugar en 1959 a los modelos ADELINE y MADELINE creados en Standford por Bernard Widrow y Marcian Hoff, ADELINE fue desarrollada para reconocer patrones binarios en una serie de bits, siendo capaz de predecir el siguiente bit en un secuencia enviada por teléfono, mientras que MADELINE fue la primera inteligencia artificial con uso real la cual utilizaba un filtro adaptativo para eliminar echo en una linea telefónica.

Debido a la falta de potencia en los sistemas de cómputo de ese tiempo y a varias dificultades que se dieron cuando se trato de expandir esta tecnologia, poco a poco se fue retirando el presupuesto para este tipo de proyectos y no hubo mucho avance hasta 1982 donde en una conferencia sobre redes neuronales cooperativas y competitivas, Japón anuncio avances con el objetivo de lograr desarrollar inteligencias artificiales más avanzadas, esto dio lugar a una competencia por parte de los investigadores americanos lo cual se tradujo en más presuspuesto y árticulos.

Finalmente, en 1986 tres grupos de investigadores independientes se enfocaron en generalizar el ajuste de pesos de una red neuronal para múltiples capas dando como resultado los algoritmos que hoy conocemos como "backpropagation", las redes neuronales que implementaban estos métodos resultaban en un aprendizaje más lento, que normalmente llevaba miles de iteraciones, pero el modelo era mucho más preciso que los previamente desarrollados.

Hoy en día las inteligencias artificiales tienen muchas aplicaciones y se estan volviendo una tecnologia fundamental en muchas áreas, con el fuerte desarrollo de los sistemas de cómputo actuales, ideas modernas y competitividad a la hora de encontrar soluciones novedosas, no cabe duda que son una herramienta con un potencial increible.

## ¿Que tipos de redes neuronales existen?

Como hemos visto las redes neuronales vienen en muchas formas y arquitecturas, y elegir el tipo de red neuronal para cada trabajo es un paso crucial cuando se desea diseñar un modelo efectivo, se presenta entonces a continuación una recopilación no exhaustiva de los diferentes tipos de redes neuronales:

### Redes neuronales prealimentada (feed-forward)

Son el tipo de red neuronal más simple donde las conexiones entre los nodos no forman un ciclo y la información fluye de las capas de entrada a potencialmente una serie de capas ocultas finalizando en la capa de salida, algunas redes neuronales que caen en esta descripción son las siguientes:

- Perceptrón: la más sencillas de las redes neuronales prealimentadas y en general una de las redes neuronales más simples, los perceptrones son algoritmos de clasificación binaria donde a partir de la entrada la cual normalmente es un vector de datos, se toma la decisión si este vector pertenece o no a una clase en específico.

- Autoencoder: también llamado autoasociador o red Diabolo, es similar a un perceptrón multicapa, con la excepción de que la última capa tiene la misma cantidad de neuronas que la capa de entrada, así, los autoencoders son modelos de aprendizaje no supervisado que tipicamente se usan en los contextos de reducción de dimensionalidad y modelos generativos.

- Probabilistica: las redes neuronales probabilisticas (PNN) son redes de cuatro capas las cuales son, una capa de entrada, una capa oculta de patrones, una capa oculta de sumación y la capa de salida, derivadas de las redes Bayesianas y utilizadas para problemas de clasificación y reconocimiento de patrones, utilizan algoritmos que aproximan las funciones de densidad de probabilidad de clases y la regla de Bayes para asignar la probabilidad de clase de cada entrada à la clase con la mayor probabilidad posterior.

- Convolucional: las redes neuronales convolucionales (CNN) es un tipo de red neuronal profunda, es decir estan compuestas de una o más capas convolucionales y múltiples capas ocultas densamente conexas entre sí. Inspiradas en el funcionamiento de cortex visual en el cerebro humano, estos modelos se aprovechan de la fuerte correlación espacial local en los datos y por lo tanto son extremadamente efectivos cuando se manejan datos dos dimensionales como imagenes o series de tiempo.

### Redes neuronales recurrentes (RNN)

Este tipo de redes destacan por que a diferencia de las anteriores propagan la información hacia atras además de hacia adelante, enviandola de etápas de procesamiento más avanzadas de vuelta a las etápas posteriores, volviendolas particularmente efectivas cuando se desea analizar secuencias de datos.

- Completamente recurrente: esta arquitectura desarrollada en la década de los 80's es una red con conexiones dirigidas entre cada par de unidades, cada conexión con un peso modificable. Ejemplos de esta arquitectura son la red de Hopfield donde todas las conexiones son simétricas y las máquinas de Boltzman que puede ser interpretada como una red de Hopfield con ruido. Sin embargo este tipo de redes son fuertemente susceptibles al error de desvanecimiento de gradiente, el cual otros tipos de redes suelen tratar de evitar.

- Simplemente recurrente: son redes neuronales sencillas (perceptrones) con la adición de una serie de unidades contextuales en la capa de entrada, las neuronas en las capas ocultas contienen conexiones a estas unidades contextuales con un peso fijo de uno, dejando una "huella" de la información que la red procesó con anterioridad.

- Memoria a corto largo plazo (LSTM): una de los tipos más populares cuando se trata de analizar secuencias de datos donde existe considerable tiempo entre cada entrada así como una mezcla entre componentes de baja y alta frecuencia, en una red LSTM las neuronas contienen un componente adicional llamada la "función de olvido", la cual decide en que momento olvidar la información que la neurona proceso con anterioridad, esta capacidad de "recordar" por periodos largos de tiempo las extremadamente versátiles.

### Modulares

Otra idea popular es la de utilizar no uno sino varios modelos cuando se desea tener una mayor certidumbre en los resultados, estas siguientes arquitecturas exploran esa posibilidad.

- Comité de máquinas (CoM): consiste en una serie de redes neuronales con sus conexiones terminales conectadas de tal manera que cada una "vota" por un resultado, este tipo de sistemas ha demostrado resultados mucho más prometedores que sus redes individuales debido a que tiende a ser mucho más resiliente ante problemas como mínimos locales y el sobreajuste de los datos.

- Asociativas: (ASNN): similar a los comités de máquinas, las redes neuronales asociativas son un conjunto de redes neuronales combinadas con la técnica de k vecinos cercanos (k-NN), esto corrige el sesgo del conjunto de las redes además de proporcionar una memoria que coincide con el conjunto de datos de entrenamiento, esto es, si se disponen de nuevos datos la red inmediatamente incrementa su habilidad predictiva sin necesidad de un reentrenamiento, a esto le llamamos autoaprendizaje.

## ¿Como funcionan?

Hasta ahora hemos visto el desarrollo y aplicaciones de esta tecnologia y hemos explorado algunos tipos de redes neuronales, no obstante, es bastante común pensar en estos sistemas como una caja negra y casi como "magia" nos damos a la tarea entonces de demistificar estas ideas con sus fundamentos matemáticos.

# Las matemáticas en las redes neuronales.

Como hemos visto en secciones anteriores el funcionamiento interno de las redes neuronales esta fuertemente fundamentado en el contexto matemático que se llevo a cabo hace décadas, es decir, las ideas y conceptos que se desarrollaron tiempo antes de su implementación más contemporanea y sin esta teoría no existiria la tecnología a la que estamos tan acostumbrados. Comenzamos entonces por desarrollar un poco de la matemática detras del bloque fundamental por el que estan compuestos estas redes: El nodo o neurona.

## Neuronas

Detalles, esquemas y diagramas de flujo

# Ejemplificación de problemática.

En este documento se trabaja con la problemática de detectar el tipo (benigno o maligno) de celulas potencialmente cancerígenas, los datos utilizados fueron encontrados en la pagina Kaggle en el siguiente [vínculo](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) y provienen de la digitalización de imagenes obtenidas en el centro de ciencias clínicas en la universidad de Wisconsin, consisten en diez característicos de las células observadas además de un número de serie (ID) y el diagnóstico de la célula (benigno o maligno). Un repaso de los atributos es el siguiente:

1. Número ID
2. Diagnóstico
3. Radio (media de la distancia del centro a los puntos del perímetro)
4. Textura (desviación estándar de valores de escala de grises)
5. Perímetro
6. Área
7. Suavidad (variación local en el los radios)
8. Compacidad (perímetro^2/área - 1)
9. Concavidad (severidad de las porciones cóncavas)
10. Puntos de concavidad (Número de porciones cóncavos en el contorno)
11. Simetría
12. Dimensión fractal ("Aproximación de la linea de playa" - 1)

El conjunto de datos tambíen otorga varias medias, desviaciones estándar entre otros descríptivos, pero no seran usados en el entrenamiento.

## Objetivo

Describir matemáticamente un modelo de clasificación binaria que haga predicciones acerca de si los datos de entrada corresponden al núcleo de una célula pertence a la clase de núcleos de células benignas o malignas.

---

## Esquema de red neuronal a implementar

Tipo

Esquema

## Implementación matemática

Implementación y detalles

## Resultados previstos

Resultados hipotéticos
