# predictPokemon ( Cloud Function python)

## Install Cloud SDK 

https://cloud.google.com/sdk/install?hl=ja

## Init Cloud SDK

```
./google-cloud-sdk/install.sh
gcloud init
gcloud components update
```

## Upload model file to Cloud Storage

Get URI of uploaded file
```
gs://kagawa-ai-lesson/model_8.hdf5
```

## deploy Cloud function

runtime オプション、memory オプションは２回目以降省略可能とのこと
```
cd predictPokemon/predictPokemon
gcloud functions deploy predictPokemon --runtime python37 --trigger-http --memory 1024MB
```
