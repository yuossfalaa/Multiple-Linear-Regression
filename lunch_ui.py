import gradio as gr

import Multi_Linear_Model
import Multiple_Linear_Regression
import Polynomial_model
from Model import Model

with gr.Blocks() as demo:
    with gr.Row():
        model_type = gr.Dropdown(["Multi Liner Regression", "Multi Polynomial Regression"],
                                 value="Multi Polynomial Regression", label="Algorithm Type", interactive=True)
        model = Model()


        def train_and_save(model_type):
            if model_type == "Multi Liner Regression":
                model = Multi_Linear_Model.Train()
                return Multiple_Linear_Regression.show_history(model)
            elif model_type == "Multi Polynomial Regression":
                model = Polynomial_model.Train()
                return Multiple_Linear_Regression.show_history(model)
            else:
                gr.Error(" ")


        def load_model(model_type):
            if model_type == "Multi Liner Regression":
                if model.can_load("multi_liner_model.json"):
                    model.load("multi_liner_model.json")
                    return Multiple_Linear_Regression.show_history(model)
                else:
                    gr.Error("Train a model first")
            elif model_type == "Multi Polynomial Regression":
                if model.can_load("polynomial_model.json"):
                    model.load("polynomial_model.json")
                    return Multiple_Linear_Regression.show_history(model)
                else:
                    gr.Error("Train a model first")
            else:
                gr.Error(" ")


        train_btn = gr.Button("Train & Save")
        load_btn = gr.Button("Load Saved Model")
    outputimage = gr.Image(default=None,height=350)

    with gr.Row():
        with gr.Column():
            textbox = gr.Textbox(label="Text",interactive=True,
                                  placeholder="Type your sentence here",
                                  value="Welcome back, Master. What can I do for you today?")
            textbox1 = gr.Textbox(label="Text",interactive=True,
                                   placeholder="Type your sentence here",
                                   value="Welcome back, Master. What can I do for you today?")
            textbox2 = gr.Textbox(label="Text",interactive=True,
                                   placeholder="Type your sentence here",
                                   value="Welcome back, Master. What can I do for you today?")
        with gr.Column():
            predict_btn = gr.Button("Predict Value")
            textbox4 = gr.TextArea(label="Text", interactive=False,
                                 placeholder="Predicted Value",)
    train_btn.click(train_and_save,inputs=[model_type], outputs=[outputimage])
    load_btn.click(load_model,inputs=[model_type], outputs=[outputimage])

if __name__ == "__main__":
    demo.launch(show_api=True)
