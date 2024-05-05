# ai-artwork-detector

## Accessing the live version of the app:
This app is live on streamlit: https://ai-generated-image-detector.streamlit.app

Please Note:
When using the ResNet model on the cloud version of the app, you might encounter a "request timeout" issue. This is due to the substantial size of the ResNet model file, which is over 200MB. The free tier of Streamlit cloud services has limitations that can lead to timeouts when loading large models like ResNet.

Please note that this timeout is a normal occurrence given the constraints of the cloud service and does not reflect any issues with the underlying code, which you can review on the linked GitHub repository. The application runs perfectly fine locally, where such constraints are not present.

## Deploy project locally
Please use this command to deploy the project locally:
```rb
streamlit run app.py --server.enableXsrfProtection false --client.showErrorDetails false
```
