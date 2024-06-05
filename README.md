# chilean-humor [Work in progress]

Tracking the history of Chilean humor.

Visit the website [here](https://db.datarisas.cl/).

## Screenshots

A list of every humor routine in the history of the [Festival de Viña](https://es.wikipedia.org/wiki/Festival_Internacional_de_la_Canci%C3%B3n_de_Vi%C3%B1a_del_Mar).
![routines](/images/website-capture-01.PNG)

A list of [Dino Gordillo](https://es.wikipedia.org/wiki/Dino_Gordillo)'s shows.
![routines-dino](/images/website-capture-02.PNG)

A transcript of [Luis Slimming](https://es.wikipedia.org/wiki/Luis_Slimming) show.
![transcript](/images/website-capture-03.PNG)

## Quick insights

A chart showing the age of each comedian when they performed on stage. Notice the trend of younger comedians starting in the 2010s (perhaps the stand-up comedy effect?).

![age](/images/age_line_plot.png)

And here is a cumulative count of how many times a man or woman has performed comedy at Festival de Viña.

![gender](/images/gender_line_plot.png)

Code: [Jupyter Notebook](/notebooks/00_explore_ficvn_data.ipynb)

## Technical explanation

I’m using [GitHub Actions](https://github.com/features/actions) and [Google Cloud Run](https://cloud.google.com/run) to automatically publish data and deploy it as a public website every time the data is updated. The website can be queried using [Datasette](https://datasette.io/).

This project is inspired by [Simon Willison's example](https://simonwillison.net/2020/Jan/21/github-actions-cloud-run/).

### Additional steps to make it work

- First, you must [enable billing](https://stackoverflow.com/questions/68536433/unable-to-submit-build-to-cloud-build-due-to-permissions-error) in your Google Cloud project.
- Follow Simon's [steps](https://simonwillison.net/2020/Jan/21/github-actions-cloud-run/), especially for setting up the Google Cloud Service Account.
- [Enable](https://cloud.google.com/endpoints/docs/openapi/enable-api) the `Cloud Build` and `Cloud Run Admin` API's in your project.
- After deploying your first Google Cloud Run service, make sure to allow unauthenticated invocations.

![invocations](/images/google-run.PNG)
