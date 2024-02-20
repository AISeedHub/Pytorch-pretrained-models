# from scripts.train import train_model

import gradio as gr
from configs.configs import Config, ArchitectSettings, CapsSettings, DatasetSettings, TrainingSettings
from scripts.train import train
from scripts.predict import predict
from models.wrappers.Detector import MODEL as DETECTOR_MODEL
from models.wrappers.Classifier import MODEL as CLASSIFIER_MODEL
from models.wrappers.Segment import MODEL as SEGMENT_MODEL
import os

MODEL_LIST = list(CLASSIFIER_MODEL.keys()) \
            + list(DETECTOR_MODEL.keys()) \
            + list(SEGMENT_MODEL.keys())

title = "Application Demo "
description = "# A Demo of Wrapping Capsule Networks"
example_list = [["./examples/" + example] for example in os.listdir("examples")]
  
with gr.Blocks() as demo:
    demo.title = title
    gr.Markdown(description)
  
    with gr.TabItem(label="Train"):
        with gr.Group():
            train_button = gr.Button(value="Train")
            text_report = gr.Textbox(label="Report")

        text_backbone = gr.Dropdown(label="Backbone", choices=MODEL_LIST)
        ckb_model = gr.CheckboxGroup(
            ["Is Full", "Is Freeze", "Is Caps"],
            label="Select Architect Options",
            value=["Is Caps"]
        )
        with gr.Accordion(label="Caps Settings", open=False):
            radio_cap_mode = gr.Radio(label="Mode", choices=[1, 2, 3, 4], value=1)
            num_cap_dims = gr.Number(label="Cap Dims", value=4, maximum=10, minimum=1, step=1)
            radio_cap_routing = gr.Radio(label="Routing", choices=["em", "dynamic", "fuzzy"], value="dynamic")
            num_cap_iter = gr.Number(label="Iteration", maximum=10, minimum=1, step=1, value=3)
            num_cap_lambda = gr.Slider(label="Lambda Value", minimum=0.0, maximum=1.0, step=0.01, value=0.01)
            num_cap_fuzzy = gr.Slider(label="Fuzzy", minimum=0.0, maximum=3.0, step=0.01, value=1.5)
        with gr.Accordion(label="Dataset Settings"):
            text_data_name = gr.Dropdown(label="Name", choices=["CIFAR10", "affNist", "Mnist", "SmallNorb", "PennFudanDataset", "LungCTscan"])
            text_data_path = gr.FileExplorer(label="Load Custom dataset", root_dir="./datasets")
            num_workers = gr.Number(label="Number of Workers", minimum=0, maximum=16, step=4)
            num_batchsize = gr.Number(label="Batch Size", minimum=1, value=4, step=4)
        with gr.Accordion(label="Training Settings"):
            ckb_gpus = gr.CheckboxGroup(label="GPU IDs", choices=[0, 1, 2, 3, 4], value=[0])
            radio_loss = gr.Radio(label="Loss", choices=["ce", "bce", "mse", "none"], value="ce")
            ckb_metrics = gr.CheckboxGroup(label="Metrics", choices=["accuracy", "map", "f1", "dice"], value=["accuracy"])
            text_ckpt = gr.Textbox(label="Checkpoint Path", value="./checkpoints")
            text_log_dir = gr.Textbox(label="Log Directory", value="./lightning_logs")
            num_epochs = gr.Number(label="Max Epochs", minimum=1, value=10)
            radio_optim = gr.Radio(label="Optimizer", choices=["adam", "sgd"], value="adam")
            radio_lr_schedule = gr.Radio(label="LR Scheduler", choices=["step", "plateau", "multistep"], value="step")
            num_lr = gr.Slider(label="Learning Rate", minimum=0.0, maximum=1.0, step=0.0001, value=0.001)
            is_early_stopping = gr.Checkbox(label="Early Stopping", value=False)
            radio_monitor = gr.Radio(label="Monitor", choices=["val_loss", "val_accuracy", "val_map"], value="val_accuracy")

    with gr.TabItem(label="Predict"):
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(label="Default Model", choices=MODEL_LIST)
            with gr.Column():
                ckpt_model = gr.File(label="load from ckpt", file_types=[".ckpt"])
        im = gr.Image(type="pil", label="image", interactive=True)
        predict_btn = gr.Button(value="predict")
        gr.Examples(examples=example_list, inputs=[im], outputs=[])

    def train_model_with_config(text_backbone, ckb_model, 
                    radio_cap_mode, num_cap_dims, radio_cap_routing, num_cap_iter, num_cap_lambda, num_cap_fuzzy, 
                    text_data_name, text_data_path, num_workers, num_batchsize, 
                    ckb_gpus, radio_loss, ckb_metrics, text_ckpt, text_log_dir, num_epochs, radio_optim, 
                    radio_lr_schedule, num_lr, is_early_stopping, radio_monitor):
        
        caps = CapsSettings(
            mode=radio_cap_mode,
            cap_dims=num_cap_dims,
            routing=radio_cap_routing,
            iteration=num_cap_iter,
            lambda_val=num_cap_lambda,
            fuzzy=num_cap_fuzzy
        )
        architect = ArchitectSettings(
            backbone=text_backbone,
            is_full="Is Full" in ckb_model,
            is_freeze="Is Freeze" in ckb_model,
            is_caps="Is Caps" in ckb_model,
            caps=caps
        )
        dataset = DatasetSettings(
            name=text_data_name,
            path=text_data_path[0] if len(text_data_path) > 0 else "",
            num_workers=num_workers,
            batch_size=num_batchsize
        )
        training = TrainingSettings(
            gpu_ids=ckb_gpus,
            loss=radio_loss,
            metrics=ckb_metrics,
            ckpt_path=text_ckpt,
            log_dir=text_log_dir,
            max_epochs=num_epochs,
            optimizer=radio_optim,
            lr_scheduler=radio_lr_schedule,
            lr=num_lr,
            early_stopping=is_early_stopping,
            monitor=radio_monitor
        )
        config = Config(
            architect_settings=architect,
            dataset_settings=dataset,
            training_settings=training
        )

        train(config)

        return f" Training completed, model saved in {text_ckpt} \n \
                run cmd: tensorboard --logdir={text_log_dir}/ to see the training process."


    train_button.click(train_model_with_config, 
                    inputs=[text_backbone, ckb_model, 
                    radio_cap_mode, num_cap_dims, radio_cap_routing, num_cap_iter, num_cap_lambda, num_cap_fuzzy, 
                    text_data_name, text_data_path, num_workers, num_batchsize, 
                    ckb_gpus, radio_loss, ckb_metrics, text_ckpt, text_log_dir, num_epochs, radio_optim, 
                    radio_lr_schedule, num_lr, is_early_stopping, radio_monitor], 
                    outputs=[text_report])
    
    def predict_model(im, model, ckpt_model):
        
        return predict(im, model, ckpt_model)
    
    predict_btn.click(predict_model, inputs=[im, model, ckpt_model], outputs=[im])
    
    
if __name__ == "__main__":
   
    # Define the Gradio interface

    demo.launch()