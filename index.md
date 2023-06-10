<head>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        inlineMath: [['$','$']]
      }
    });
  </script>
  <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
</head>

# Redes neuronales

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

![AlphaZeroLogo](.\rsc\AlphaZeroLogo.png)

Este software desarrollado por Google se centra aprender a jugar partidas de juegos generales, principalmente ajedrez, go y shogi, a un nivel donde es capaz de ganar con facilidad incluso a otros sistemas de aprendizaje automático y redes neuronales diseñadas para destacar específicamente en esos juegos. Consiste en una combinación entre una red neuronal de tipo convolucional y un algoritmo de búsqueda de árbol de Monte Carlo, esta combinación le permite aprender de manera muy rápida jugando partidas contra si mismo y alcanzar un nivel profesional en un periodo muy corto de tiempo.

Dall-E 2, Transformer

![Dall-ELogo](.\rsc\Dall-ELogo.webp)

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

## Nodo

El nodo o neurona es, por decirlo de alguna manera, la unidad más simple de una red neuronal, en la red los nodos están conectados entre sí de manera jerárquica, donde la salida de algunos nodos es la entrada de otros, podemos entender de manera burda un nodo como un objeto o función que recibe una serie de números reales como entrada, realiza alguna operación y devuelve un número real como salida.

Los nodos se componen de los siguientes elementos:

- Una función de activación $f:\mathbb{R}\rightarrow \mathbb{R}$ , la cual usualmente es la misma para todos los nodos excepto, tal vez, para los nodos que pertenecen a la capa de salida (la cual veremos màs adelante)

- Un peso $w_{j}\in\mathbb{R}$ asociado a cada nodo que se encuentra conectado a su entrada

- Un sesgo $b\in\mathbb{R}$ asociado al nodo en cuestión

Antes de ahondar un poco en el uso y funcionamietno de cada uno de estos componentes es importante mencionar lo que son las capas de una red neuronal.

## Capas

En la mayoria de las redes neuronales, los nodos son agrupados en capas, una capa es una serie de nodos (comúnmente representado como un vector) las cuales no contienen conexiones entre sí, una red neuronal consiste en una serie ordenada de capas cuyos nodos estan conectados de una capa a otra, llamamos a la capa _densa_ si todo nodo de dicha capa contiene una conexión a cada uno de los nodos de la capa siguiente.

En el siguiente diagrama se ilustra un esquema de una red neuronal sencilla con tres capas, tres nodos en su primera capa, cuatro en su segunda y uno en su tercera.

![DiagramaRedNeuronal](.\rsc\DiagramaRed1.png)

Además distinguimos la primera capa como la capa de entrada cuyos nodos reciben como entrada el vector de datos que otorgamos a la red neuronal, y llamamos a la última capa la capa de salida, la salida de los nodos de esta capa forma el vector que interpretaremos como el resultado que otorga la red.

## Funcionamiento de cada nodo

Como vimos con anterioridad los nodos se componen esencialmente de: una función de activación $f$ compartida entre todos los nodos (excepto en la capa de salida), un peso $w_{j}$ para cada nodo conectado a su entrada y un sesgo $b$ específico al nodo. Pasaremos a ver como cada noda transforma sus entradas en su salida.

Análizemos primero la función de acivación, llamamos $f:\mathbb{R}\rightarrow\mathbb{R}$ a la función de activación, esta es la función la cual cada nodo utilizará para modificar la entrada de correspondiante, es decir la sálida del nodo está determinada por $f\left(z\right)$, la elección de la función de activación es de suma importancia para el desempeño de la red neuronal, existen una variedad de funciones y depende de la arquitectura de la red así como del contexto del problema cual debe usarse, ultimadamente, lo único que necesitamos de esta función es que sea derivable en casi todos sus puntos, pero es común tomar en cuenta otras características como simetría, linealidad, concavidad, acotamientos, etc. Aquí hay algunos ejemplos de funciones de activación comunes:

![FuncionesDeActivación](.\rsc\FuncionesActivacion.jpeg)

Continuamos con los pesos y sesgos. Sea $n_{i}^{l}$ el $i$-ésimo nodo de la $l$-ésima capa, asociamos a este nodo un peso $w_{ij}\in\mathbb{R}$ por cada $j$-ésimo nodo de la $\left(l-1\right)$-ésima capa anterior, y un sesgo $b_{i}\in\mathbb{R}$. La salida $s_{i}^{l}$ de este nodo está determinada por $s_{i}^{l} = f\left(z\right)$ donde $f$ es la función de activación y

$$
z = b_{i} + w_{i1}s_{1}^{(l-1)} + w_{i2}s_{2}^{(l-1)} + \cdots + w_{im_{(l-1)}}s_{m_{(l-1)}}^{(l-1)} = b_{i} + \sum_{j = 1}^{m_{(l-1)}} w_{ij} s_{j}^{(l-1)}
$$

Aquí $s_{j}^{(l-1)}$ es la salida del $j$-ésimo nodo de la $(l-1)$-ésima capa anterior y la cual suponemos que contiene $m_{(l-1)}$ nodos.

El siguiente diagrama ejemplifica lo anterior.

![DiagramaNodo](.\rsc\DiagramaRed2.png)

En síntesis, se toma el producto de cada entrada del nodo con su peso correspondiente, despues se suman estos valores y se le agrega el sesgo para obtener $z$, finalmente se evalua la función de activación en $z$ para obtener $s_{i}^{l}$ la salida del nodo.

## Propagación hacia adelante

Ahora veremos la descripción matemática de la propagación completa de los datos desde la capa de entrada hasta la de salida. Consideremos la siguiente red neuronal:

- $P$ capas

- $m_{l}$ nodos en su $l$-ésima capa

- Una función de activación $f:\mathbb{R}\rightarrow\mathbb{R}$ la cual supondremos que es la misma para cada nodo

- Todas las capas son densas es decir la salida de cada nodo $n_{j}^{l}$ forma parte de la entrada de cada nodo $n_{i}^{(l+1)}$

- Cada nodo $n_{i}^{l}$ que no pertenesca a la capa de entrada tiene asociados unos pesos $w_{ij}^{l}$ y un sesgo $b_{i}^{l}$

Definimos entonces la matriz de pesos para la $l$-ésima capa como:

$$
W^{l} = \left(w_{ij} \right)_{i=1, j=1}^{m_{l}, m_{(l-1)}} = 
\begin{bmatrix}
    w_{11} & w_{12} & \cdots & w_{1m_{(l-1)}} \\
    w_{21} & w_{22} & \cdots & w_{2m_{(l-1)}} \\
    \vdots & \vdots & \ddots & \vdots \\
    w_{m_{l}1} & w_{m_{l}2} & \cdots & w_{m_{l}m_{(l-1)}}
\end{bmatrix}
$$

A su vez definimos el vector de salidas y el vector de sesgos de la $l$-ésima capa como sigue:

$$
s^{l} =
\begin{bmatrix}
    s_{1}^{l} \\
    s_{2}^{l} \\
    \vdots \\
    s_{m_{l}}^{l}
\end{bmatrix}
, 
b^{l} =
\begin{bmatrix}
    b_{1}^{l} \\
    b_{2}^{l} \\
    \vdots \\
    b_{m_{l}}^{l}
\end{bmatrix}
$$

De esta manera la propagación hacia adelante queda definida por las siguientes dos ecuaciones:

$$
z^{l} = W^{l}s^{(l-1)} + b^{l} \\
s^{l} = f\left(z^{l}\right)
$$

Al momento de la implementación estas operaciones deben de realizarse una vez por capa, el vector $s^{P}$ es el resultado que devuelve nuestra red neuronal.

## Descenso de gradiente y propagación hacia atras

Ahora queda describir como se lleva a cabo el aprendizaje, como vimos, los pesos y el sesgo son los parámetros fundamentales que se modificarán al momento del entrenamiento, la cuestión ahora es ¿como se modifican para que nuestra red neuronal tenga un buen desempeño? En el entrenamiento supervisado la idea es reducir la diferencia entre los resultados que se obtienen y los que esperamos, así debemos construir una función de pérdida.

La función de pérdida debe de ser una medida de dismilitud, la cual queremos minimizar, la elección de esta medida es crucial para el desempeño de nuestra red y depende fuertemente del uso que se le quiera dar y la problemática a resolver, aquí hay algunas funciones que se utilizan comúnmente (en especial para problemas de clasificación):

- Distancia $\mathcal{L}_{1}$

$$
||y-o||_{1}
$$

- Distancia $\mathcal{L}_{2}$

$$
||y-o||_{2}^{2}
$$

- Pérdida de Chevyshev

$$
\max_{j}|\sigma\left(o\right)^{(j)}-y^{(j)}|
$$

- Pérdida de Hinge

$$
\sum_{j}\max\left(0,\frac{1}{2}-{y}^{(j)}o^{(j)}\right)
$$

- Pérdida de entropía cruzada logarítmica

$$
-\sum_{j}y^{(j)}\log\sigma\left(o\right)^{(j)}
$$

donde $y$ es el etiquetado verdadero codificado como $1$ o $0$ y $o$ son los valores devueltos por la red y $\sigma$ denota la probabilidad estimada (en la pérdida de hinge las etiquetas son de $-1$ o $1$).

A las funciones de pérdida les pedimos que sean diferenciables en casi todos sus puntos, es justamente el gradiente lo que nos ayudará a ajustar los pesos y sesgos de la red neuronal. Recordemos que buscamos encontrar el minimizar el error entre las etiquetas predecidas y las reales, así tenemos en nuestras manos un problema de cálculo multivariable.

Nuestra función de error puede verse como una función $\mathcal{J}\left(w, b, x,y\right)$ donde $w$ y $b$ son los pesos y sesgos de la red respectivamente, $x$ son los datos de entrada que otorgamos a la red y $y$ es la etiqueta real que le corresponde a los datos. Supongamos que disponemos de $N$ pares $(x_{t},y_{t})$ con $t = 1, 2, \dots, N$ de datos y etiquetas, entonces el error de nuestra red despues de propagar cada dato a traves de ella se expresa:

$$
\mathcal{J}\left(w,b\right) = \frac{1}{N}\sum_{t=1}^{N}\mathcal{J}\left(w,b,x_{t},y_{t}\right)
$$

Y con esto podemos describir el descenso de gradiente para cada peso $w_{ij}^{l}$ y cada sesgo $b_{i}^{l}$ como sigue:

$$
\begin{align*}
  w_{ij}^{l} &= w_{ij}^{l}-\alpha\frac{\partial}{\partial w_{ij}^{l}}\mathcal{J}\left(w,b\right) \\
  b_{i}^{l} &= b_{i}^{l}-\alpha\frac{\partial}{\partial b_{i}^{l}}\mathcal{J}\left(w,b\right)
\end{align*}
$$

Donde $\alpha\in\mathbb{R}$ es un paramétro al que llamamos el _paso_ el cual controla que tanto los pesos se ajustaran en la dirección que marca el gradiente (cabe mencionar el ligero abuso de notación para describir que actualizaremos los pesos).

Queda pendiente el cálculo de las derivadas parciales de la función de pérdida con respecto a los pesos y sesgos, ya que queda mejor ejemplificado con una elección concreta de función de pérdida y función de activación, en la parte dos de este documento, se escogerán estas funciones y se realizara el ajuste de pesos y sesgos paso a paso.

## Conclusión

Como hemos visto, se requiere un desarrollo matemático extenso para entender el funcionamiento de las redes neuronales incluso para los casos más sencillos como el que vimos con anterioridad, existen muchos tipos de redes y arquitecturas distintas y cada una realiza alteraciones en menor o mayor grado al esquema que describimos, no obstante podemos resumir su funcionamiento con el siguiente diagrama de flujo:

![DiagramaFlujoRed](.\rsc\DiagramaRed3.png)

# Ejemplificación de problemática.

En este documento se trabaja con la problemática de detectar el tipo (benigno o maligno) de celulas potencialmente cancerígenas, los datos utilizados fueron encontrados en la pagina Kaggle en el siguiente [vínculo](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) y provienen de la digitalización de imagenes obtenidas en el centro de ciencias clínicas en la universidad de Wisconsin, consisten en diez característicos de las células observadas además de un número de serie (ID), el diagnóstico de la célula (benigno o maligno) y un total de 569 entradas. Un repaso de los atributos es el siguiente:

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

## Red neuronal propuesta

Cómo método de implementación del modelo, se propone una red neuronal perceptron multicapa, con un solo nodo en la capa de salida que nos indique si los datos pertenecen a la clase de núcleos de células benignos ($0$) o cancerígenos ($1$), describimos la red neuronal como sigue:

- $3$ capas, una de entrada, una capa oculta y una de salida

- $10$ nodos en la capa de entrada, una por cada variable independiente

- $8$ nodos en la capa oculta

- un único nodo en la capa de salida el cual interpretamos como el resultado de nuestra red (una estimación de la probabilidad de que los datos sean clasificados como cancerígenos)

las capas son densas entre sí con todos sus pesos y sesgos modificables a la hora del entrenamiento de la red, además:

- utilizamos como función de activación la función sigmoide $\frac{1}{1+e^{-x}}$ para todos los nodos de la red neuronal

- utilizamos como función de pérdida la suma (normalizada) de la diferencia de cuadrados entre $y^{t}$ la clasificación real de la $t$-ésima entrada y $s^{3}\left(x^{t}\right)$ la salida de la última capa dada la $t$-ésima entrada, escribimos entonces:

$$
\mathcal{J}\left(w,b,x^{t},y^{t}\right) = \frac{1}{2}||y^{t}-s^{3}\left(x^{t}\right)||^{2}
$$

Lo anterior para cada par $\left(x^{t},y^{t}\right)$ de datos/clasificación, deseamos minimizar la función de pérdida para todos estos paresm suponiendo $N$ pares, tenemos que:

$$
\begin{align*}
  \mathcal{J}\left(w,b\right) &= \frac{1}{N}\sum_{t=1}^{N}\frac{1}{2}||y^{t}-s^{3}\left(x^{t}\right)||^{2} \\
  &= \frac{1}{N}\sum_{t=1}^{N}\mathcal{J}\left(w,b,x^{t},y^{t}\right)
\end{align*}
$$

## Implementación matemática

### Preprocesamiento y consideraciones

Es necesario mencionar las modificaciones que hacemos al conjunto de datos para su correcto manejo con estas tecnologías, primeramente debemos verificar la integridad de los datos para ver si es necesaria alguna imputación o descarte de los mismos, no obstante, el conjunto de datos seleccionado ya tiene una calidad alta y no hay datos faltantes.

Continuamos con la manipulación, descartando las columnas que no se van a utilizar, la medias y desviaciones estandar que acompañan a los datos, despues transformamos la variable indicadora de diabetes de categórica a numérica (binaria) asociando "B" con "$0$" y "M" con "$1$" y la separamos de nuestros datos quedando como un vector $y$.

Otro paso importante es la normalización de los datos, para un mejor entrenamiento y mejorar la tasa de convergencia de la red hacia el punto mínimo de la función de pérdida, utilizamos un proceso de normalización min-max (que en realidad es solo un escalamiento) el cual transforma todos nuestro datos numéricos a un rango entre $0$ y $1$ de manera lineal:

$$
x_{\text{nueva}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}
$$

Ahora separamos nuestros datos en dos conjuntos, el $80$% ($456$ entradas) de los datos (con sus respectivas etiquetas) se utilizaran en el entrenamiento, y el $20$% ($113$ entradas) restante en la validación hipotética del modelo.

Finalmente al tratárse de un problema de clasificación que involucra un diagnóstico médico es muy probable que el conjunto de datos presente un desbalance considerable entre las clases "M" y "B", una inspección de los datos revela lo anterior como cierto. Para nivelar las clases utilizaremos una técnica de remuestreo con reemplazamiento hasta que la clase con menor cantidad de datos tenga los mismos que la clase con mayor cantidad (oversampling), este remuestreo solo se aplica a los datos que son usados para el entrenamiento.

### Propagación hacia adelante

Al iniciarlizar nuestra red los pesos y sesgos son valores aleatorios y se espera que el entrenamiento subsecuente los ajuste hasta que el modelo sea funcional, ya vimos en un sección anterior que la propagación hacia adelante se ve gobernada por las siguientes ecuaciones:

$$
\begin{align*}
  z^{l} = W^{l}s^{(l-1)} + b^{l} \\
  s^{l} = f\left(z^{l}\right)
\end{align*}
$$

donde $W^{l}$ y $b^{l}$ es la matriz de pesos y el vector de sesgos de la $l$-ésima capa respectivamente, $f$ es nuestra función de activación (la función sigmoide) y $s^{l}$ es la salida de la $l$-ésima capa. Para ejemplificar realizamos el cálculo de la propagación de la capa de entrada a la capa oculta.

$$
\begin{matrix}
\begin{bmatrix}
    w_{11}^{1} & w_{12}^{1} & w_{13}^{1} & w_{14}^{1} & w_{15}^{1} & w_{16}^{1} & w_{17}^{1} & w_{18}^{1} & w_{19}^{1} & w_{110}^{1} \\
    w_{21}^{1} & w_{22}^{1} & w_{23}^{1} & w_{24}^{1} & w_{25}^{1} & w_{26}^{1} & w_{27}^{1} & w_{28}^{1} & w_{29}^{1} & w_{210}^{1} \\
    w_{31}^{1} & w_{32}^{1} & w_{33}^{1} & w_{34}^{1} & w_{35}^{1} & w_{36}^{1} & w_{37}^{1} & w_{38}^{1} & w_{39}^{1} & w_{310}^{1} \\
    w_{41}^{1} & w_{42}^{1} & w_{43}^{1} & w_{44}^{1} & w_{15}^{1} & w_{16}^{1} & w_{17}^{1} & w_{48}^{1} & w_{49}^{1} & w_{410}^{1} \\
    w_{51}^{1} & w_{52}^{1} & w_{53}^{1} & w_{54}^{1} & w_{15}^{1} & w_{16}^{1} & w_{17}^{1} & w_{58}^{1} & w_{59}^{1} & w_{510}^{1} \\
    w_{61}^{1} & w_{62}^{1} & w_{63}^{1} & w_{64}^{1} & w_{15}^{1} & w_{16}^{1} & w_{17}^{1} & w_{68}^{1} & w_{69}^{1} & w_{610}^{1} \\
    w_{71}^{1} & w_{72}^{1} & w_{73}^{1} & w_{74}^{1} & w_{15}^{1} & w_{16}^{1} & w_{17}^{1} & w_{78}^{1} & w_{79}^{1} & w_{710}^{1} \\
    w_{81}^{1} & w_{82}^{1} & w_{83}^{1} & w_{84}^{1} & w_{15}^{1} & w_{16}^{1} & w_{17}^{1} & w_{88}^{1} & w_{89}^{1} & w_{810}^{1}
\end{bmatrix}
&
\begin{bmatrix}
    s^{0}_{1} = x^{t}_{1} \\
    s^{0}_{2} = x^{t}_{2} \\
    s^{0}_{3} = x^{t}_{3} \\
    s^{0}_{4} = x^{t}_{4} \\
    s^{0}_{5} = x^{t}_{5} \\
    s^{0}_{6} = x^{t}_{6} \\
    s^{0}_{7} = x^{t}_{7} \\
    s^{0}_{8} = x^{t}_{8} \\
    s^{0}_{9} = x^{t}_{9} \\
    s^{0}_{10} = x^{t}_{10}
\end{bmatrix}
&
+
&
\begin{bmatrix}
    b^{1}_{1} \\
    b^{1}_{2} \\
    b^{1}_{3} \\
    b^{1}_{4} \\
    b^{1}_{5} \\
    b^{1}_{6} \\
    b^{1}_{7} \\
    b^{1}_{8}
\end{bmatrix}
&
=
&
\begin{bmatrix}
    z^{1}_{1} \\
    z^{1}_{2} \\
    z^{1}_{3} \\
    z^{1}_{4} \\
    z^{1}_{5} \\
    z^{1}_{6} \\
    z^{1}_{7} \\
    z^{1}_{8}
\end{bmatrix}
\end{matrix}
$$
### Descenso de gradiente y propagación hacia atras

Previamente vimos que la actualización de los pesos y sesgos por descenso de gradiente se realiza según las siguientes ecuaciones:

$$
\begin{align*}
  w_{ij}^{l} &= w_{ij}^{l}-\alpha\frac{\partial}{\partial w_{ij}^{l}}\mathcal{J}\left(w,b\right) \\
  b_{i}^{l} &= b_{i}^{l}-\alpha\frac{\partial}{\partial b_{i}^{l}}\mathcal{J}\left(w,b\right)
\end{align*}
$$

Y quedo pendiente el cálculo del los términos $\frac{\partial}{\partial w_{ij}^{l}}\mathcal{J}\left(w,b\right)$ y $\frac{\partial}{\partial b_{i}^{l}}\mathcal{J}\left(w,b\right)$, ya elegida la función de pérdida, usamos la regla de la cadena para derivadas parciales para expresar estos términos, ejemplificamos el cálculo de $\frac{\partial\mathcal{J}}{\partial w_{12}^{2}}$:

$$
\frac{\partial\mathcal{J}}{\partial w_{12}^{2}} = \frac{\partial\mathcal{J}}{\partial s^{3}_{1}}\frac{\partial s^{3}_{1}}{\partial z^{3}_{1}}\frac{\partial z^{3}_{1}}{\partial w_{12}^{2}}
$$

El último término se simplifica a:

$$
\frac{\partial z^{3}_{1}}{\partial w_{12}^{2}} = \frac{\partial}{\partial w_{12}^{2}}\left(b_{1}^{1} + \sum_{i=1}^{8}w_{1i}^{1}s_{i}^{2}\right) = s_{2}^{2}
$$

también tenemos que:

$$
\frac{\partial s}{\partial z} = f'\left(z\right)
$$

que en este caso es:

$$
\frac{\partial s}{\partial z} = f\left(z\right)\left(1-f\left(z\right)\right)
$$

Nos queda entonces el término $\frac{\partial\mathcal{J}}{\partial s_{1}^{3}}$ recordemos que nuestra función de pérdida el cuadrado de las diferencias. Hacemos $u = ||y-s^{3}(z_{1}^{3})||$ y tenemos $\mathcal{J} = \frac{1}{2} u^{2}$. Definimos:

$$
\delta_{i}^{l} = -\left(y-s_{i}^{l}\right)\cdot f'\left(z_{i}^{l}\right)
$$

Escribimos entonces la expresión completa de la derivada de la función de pérdida:

$$
\frac{\partial}{\partial W_{ij}^{l}}\mathcal{J}\left(W,b,x,y\right) = s_{j}^{l}\delta_{i}^{l+1}
$$

Pero entonces, ¿como calculamos las $\delta_{i}^{l}$ para las capas ocultas?, utilzamos la siguiente relación:

$$
\delta_{j}^{l} = \left(\sum_{i=1}^{s_{l+1}}w_{ij}^{l}\delta_{i}^{l+1}\right)f'\left(z_{j}^{l}\right)
$$

El cálculo de las derivadas queda como sigue:

$$
\begin{align*}
  \frac{\partial}{\partial W_{ij}^{l}}\mathcal{J}\left(W,b,x,y\right) &= s_{j}^{l}\delta_{i}^{l+1} \\
  \frac{\partial}{\partial b_{i}^{l}}\mathcal{J}\left(W,b,x,y\right) &= \delta_{i}^{l+1}
\end{align*}
$$

Finalmente implementamos el descenso de gradiente como sigue:

$$
\begin{align*}
  w_{ij}^{l} &= w_{ij}^{l}-\alpha\frac{\partial}{\partial w_{ij}^{l}}\mathcal{J}\left(w,b\right) \\
  b_{i}^{l} &= b_{i}^{l}-\alpha\frac{\partial}{\partial b_{i}^{l}}\mathcal{J}\left(w,b\right)
\end{align*}
$$

## Resultados previstos

Despues de una cierta cantidad de iteraciones del proceso de entrenamiento evaluamos el desempeño de nuestra red con el conjunto de validación que separamos anteriormente, si encontramos que la precisión no es aceptable repetimos con una mayor cantidad de iteraciones manteniendo en cuenta los problemas que el sobreajuste puede causar, sin embargo, la naturaleza de la problemática y la arquitectura de la red hare que el modelo sea propenzo a un alto número de falsos negativos, lo cual puede ser contraproducente para el modelo, estos problemas pueden ser detectados con otras medidas de precisión distintas a la que estamos usando o con herramientas como la matriz de confusión, podemos utilizar técnicas de nivelación del conjunto de datos más sofisticadas para tratar de superar esta dificultad y ajustar el umbral de clasificación si vemos aceptable una mayor cantidad de falsos positivos.

## Referencias

https://www.researchgate.net/publication/343554356_Mathematical_foundations_of_neural_networks_Implementing_a_perceptron_from_scratch

https://arxiv.org/abs/1702.05659

https://arxiv.org/abs/1806.07366

https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/History/history1.html

https://www.v7labs.com/blog/neural-networks-activation-functions#3-types-of-neural-networks-activation-functions