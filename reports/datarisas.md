# Datarisas

[Datarisas](https://www.datarisas.cl) es un proyecto dedicado a analizar la historia del humor en Chile. Hemos recopilado y analizado datos de miles de chistes contados sobre el escenario más importante del país: el [Festival Internacional de la Canción de Viña del Mar](https://es.wikipedia.org/wiki/Festival_Internacional_de_la_Canci%C3%B3n_de_Vi%C3%B1a_del_Mar) (un festival de música y entretenimiento que se celebra anualmente en la ciudad de Viña del Mar, Chile). Nuestro objetivo es no solo preservar estos momentos de humor, sino también ofrecer una mirada profunda y curiosa sobre cómo ha evolucionado la comedia a lo largo de las décadas.

## Datos

El sitio se alimenta de una base de datos con [**6.276** chistes](https://db.datarisas.cl/humor/jokes) contados sobre el escenario del Festival de Viña del Mar desde 1960 hasta la fecha, extraídos de transcripciones de [128 rutinas](https://db.datarisas.cl/humor/routines) de las cuales se tienen registro en Youtube.

Todo el proceso de transcripción y extracción de los chistes fue **completamente automatizado** usando **inteligencia artificial**. Por el momento ningún humano ha editado una sola coma. Si te interesa conocer como se realiza el proceso, puedes [revisar el código aquí](/src/chilean_humor/).

Esperamos en el futuro completar con los chistes de todas las rutinas y agregar los datos de otros festivales populares. Si te interesa colaborar con esta causa, [contáctanos](https://twitter.com/aastroza).

## Análisis

A continuación, se presentan algunos análisis interesantes que surgen de los datos recopilados.

El primer gráfico muestra la edad que tenía cada comediante al momento de presentarse en el escenario del Festival de Viña. Se puede apreciar una tendencia hacia comediantes más jóvenes comenzando en la década de 2010, ¿quizás esto se deba al efecto del *stand-up comedy*?

![age](/images/age_line_plot_spanish.png)

El segundo gráfico muestra un conteo acumulativo de cuántas veces un hombre o una mujer se han subido a hacer humor en el escenario del Festival de Viña a lo largo de los años.

![gender](/images/gender_line_plot_spanish.png)

## Créditos

Datarisas es un proyecto desarrollado por [Alonso Astroza](https://github.com/aastroza) y [Alfonso Concha](https://github.com/sikolio).

## Cómo citar

Si deseas utilizar los datos de este proyecto en tus propios análisis o investigaciones, por favor cita el proyecto de la siguiente manera:

Astroza, A. y Concha, A. (2024). Datarisas: Análisis de la historia del humor chileno. https://www.datarisas.cl