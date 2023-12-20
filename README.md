# picoGPT
Introduce pico (nano/1000) version of GPT built and trained from scratch on a free Google-Colab GPU. This repo showcases model creation, dataset preparation tailored to the massive limitation in computational power, training and validation processes along with inference, which enables user to specify same model hyperparameters as OpenAI API such as temperature and top_p.
## Training and Inference
The model was trained for 7 epochs, dataset and hyperparameters can be found in [config](https://github.com/AkmOleksandr/picoGPT/blob/main/config.py). Something worth mentioning is picoGPT's prediction for the following prompt: "People should know that AI" it predicted: "is a reliable tool for improving the quality of life.". Let's just hope it tells the truth😅.

Besides the example mentioned above it was dared to expand a couple more prompts:

<img width="802" alt="image" src="https://github.com/AkmOleksandr/picoGPT/assets/115898001/0650f01e-b0d3-4b2f-bdbc-ff79907af7ce">




All the examples can be also found in [Training_and_Inference](https://github.com/AkmOleksandr/picoGPT/blob/main/Training_and_Inference.ipynb) notebook.
## Download Model
The model is uploaded on HuggingFace use the following face to access it: [🤗](https://huggingface.co/AIisnotapig/picoGPT/tree/main).
