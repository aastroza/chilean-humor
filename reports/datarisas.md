# Datarisas

Bienvenido a [Datarisas](https://www.datarisas.cl), un proyecto dedicado a analizar la historia del humor en Chile. Hemos recopilado y analizado datos de miles de chistes contados sobre el escenario más importante del país. Nuestro objetivo es no solo preservar estos momentos de humor, sino también ofrecer una mirada profunda y curiosa sobre cómo ha evolucionado la comedia a lo largo de las décadas.

## Datos

El sitio se alimenta de una base de datos con [**6.276** chistes](https://db.datarisas.cl/humor/jokes) contados sobre el escenario del [Festival de Viña del Mar](https://es.wikipedia.org/wiki/Festival_Internacional_de_la_Canci%C3%B3n_de_Vi%C3%B1a_del_Mar) desde 1960 hasta la fecha, extraídos de transcripciones de [128 rutinas](https://db.datarisas.cl/humor/routines) de las cuales se tienen registro en Youtube.

Todo el proceso de transcripción y extracción de los chistes fue **completamente automatizado** usando **inteligencia artificial**. Por el momento ningún humano ha editado una sola coma. Si te interesa conocer como se realiza el proceso, puedes [revisar el código aquí](/src/chilean_humor/).

Esperamos en el futuro completar con los chistes de todas las rutinas y agregar los datos de otros festivales populares. Si te interesa colaborar con esta causa, [contáctanos](https://twitter.com/aastroza).

## Análisis

Algunas cosas interesantes que surgen de analizar rápidamente los datos.

Acá un gráfico con la edad que tenía cada comediante al momento de presentarse en el escenario. Hay una tendencia a comediantes más jóvenes comenzando en la década de 2010 (¿quizás el efecto del *stand-up comedy*?).

![age](/images/age_line_plot_spanish.png)

Y aquí hay un conteo acumulativo de cuántas veces un hombre o una mujer se han subido a hacer humor en el escenario del Festival de Viña.

![gender](/images/gender_line_plot_spanish.png)

## Créditos

Datarisas es un proyecto desarrollado por [Alonso Astroza](https://github.com/aastroza) y [Alfonso Concha](https://github.com/sikolio).