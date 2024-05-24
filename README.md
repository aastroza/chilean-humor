# chilean-humor [Work In Progress]

Tracking the history of Chilean humor.

Visit the website [here](https://chilean-humor-lomay7b72q-uc.a.run.app/).

![routines](/images/website-capture-01.PNG)

![routines-dino](/images/website-capture-02.PNG)

## Technical Explanation

I’m using GitHub Actions and Google Cloud Run to publish data and deploy it as a public website that can be queried using Datasette.

This project is inspired by [Simon Willison's example](https://simonwillison.net/2020/Jan/21/github-actions-cloud-run/).

### Some aditional tricks to make it work

- First, you must [enable billing](https://stackoverflow.com/questions/68536433/unable-to-submit-build-to-cloud-build-due-to-permissions-error) in your Google Cloud project.
- Follow Simon's [steps](https://simonwillison.net/2020/Jan/21/github-actions-cloud-run/), especially for setting up the Google Cloud Service Account.
- [Enable](https://cloud.google.com/endpoints/docs/openapi/enable-api) the `Cloud Build` and `Cloud Run Admin` API's in your project.
- After deploying your first Google Cloud Run service, make sure to allow unauthenticated invocations.

![invocations](/images/google-run.PNG)