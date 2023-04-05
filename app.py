import gradio as gr
from fastai.vision.all import *
import skimage

loder = load_learner('model.pkl')

labels = loder.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = loder.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Image Detector"
description = "Detect whether a image is table or figure"
examples = ['a069530_a69530_14684060_0001-05.jpg']
interpretation='default'
enable_queue=True

gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=3),title=title,description=description,article=article,examples=examples,interpretation=interpretation,enable_queue=enable_queue).launch()
