# GenML_WebAPI
Flask wep app that serves the purpose of an API for a mixture of Generative ML tasks with a Computational Creativity and Game AI in mind.

## Running locally

- Install library requirements from the `requirements.txt` file. 
`pip install -r requirements.txt`

- Run the Flask web app: `python app.py`


## Getting images (GET)

#### Single image.
- `/SingleImage `
    Get a single generated image in a JSON response with the following signature:
    ```
    {
        "format": "image format (.png, .jpg, etc.)",
        "img": "a string representing an image in base64 format",
        "msg": "output message",
        "size": [width, height]
    }
    ```
- `/SingleImageBrowser`
    You can open in a browser to see a generated image
#### Image batch:
- `/ImageBatch/<batch_size>`
    Get generated images in a JSON response with the following signature:
    ```
    {
        "format": "image format (.png, .jpg, etc.)",
        "imgs": ["a list of strings representing images in base64 format"],
        "msg": "output message",
        "size": [width, height]
    }
    ```

- `/ImageBatchBrowser/<batch_size>`
    You can open in a browser to see `<batch_size>` generated images

## Uploading images (POST)

#### Image batch:

Upload a set of images and get back an evolved set of images of `<batch_size>`.

Expects `multipart/form-data` with the following structure:
```
["selected_images" : [file_list],
"batch_size" : batch_size]
```

- `/ImageBatch`
    Response has the same structure as GET action for this endpoint


- `/ImageBatchBrowser`
    You can open in a browser to see `<batch_size>` generated images





