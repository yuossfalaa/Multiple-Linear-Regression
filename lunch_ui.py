import gradio as gr
import numpy as np
import Multi_Linear_Model
import Multiple_Linear_Regression
import Polynomial_model
from Model import Model


def _string_to_int_(input_str):
    try:
        result = float(input_str)
        return result
    except ValueError:
        print(f"Error: '{input_str}' is not a valid number.")
        return 0


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
    outputimage = gr.Image(default=None, height=350)

    with gr.Row():
        with gr.Column():
            feature1 = gr.Textbox(label="house age", interactive=True,
                                  placeholder="house age",
                                  )
            feature2 = gr.Textbox(label="nearest MRT station", interactive=True,
                                  placeholder="distance to the nearest MRT station",
                                  )
            feature3 = gr.Textbox(label="convenience stores", interactive=True,
                                  placeholder="number of convenience stores",
                                  )
        with gr.Column():
            predict_btn = gr.Button("Predict Value")
            Prediction = gr.TextArea(label="Output", interactive=False,
                                     placeholder="Predicted house price of unit area", )
        def predict(feature1, feature2, feature3):
            return Multiple_Linear_Regression.predict(model, np.array(
                [_string_to_int_(feature1), _string_to_int_(feature2), _string_to_int_(feature3)]))

    train_btn.click(train_and_save, inputs=[model_type], outputs=[outputimage])
    load_btn.click(load_model, inputs=[model_type], outputs=[outputimage])
    predict_btn.click(predict, inputs=[feature1, feature2, feature3],outputs=[Prediction])

if __name__ == "__main__":
    demo.launch(show_api=True)
