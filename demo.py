import gradio as gr
import numpy as np
import cv2
from PIL import Image
from types import SimpleNamespace

# Re‑use the implementation and heavyweight models that are already
# initialised inside ``instantedit.py``
import instantedit as ie  # noqa: E402, the import triggers model loading

# ``instantedit`` defines and initialises the following globals when it is
# imported: ``pipe``, ``tokenizer``, ``encoder``, and helper classes such as
# ``LocalBlend``/``AttentionControlEdit``.  The only thing we need to patch
# in at run‑time is the per‑request value for
# ``controlnet_conditioning_scale`` – the original ``inference`` helper looks
# this value up from ``instantedit.args``.  We therefore overwrite that
# attribute on every call before invoking ``instantedit.inference``.

def generate(
    image: Image.Image,
    source_prompt: str,
    edit_prompt: str,
    positive_prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    mask_threshold: float,
    controlnet_conditioning_scale: float,
    dpg_weight: float,
    cfg_weight: float,
    seed: int,
    local: str,
):
    """Run the Per‑flow Instant‑Edit pipeline and return the edited image."""
    if image is None:
        return None

    # Ensure RGB; gr.Image returns PIL.Image by default.
    img_pil = image.convert("RGB").resize((512,512))

    # Build the Canny edge map required by the ControlNet branch.
    np_img = np.array(img_pil)
    canny = cv2.Canny(np_img, 100, 200)
    control_image = Image.fromarray(canny)

    # Patch the desired conditioning scale into the imported module so that
    # ``instantedit.inference`` picks it up without modification.
    ie.args = SimpleNamespace(controlnet_conditioning_scale=controlnet_conditioning_scale)

    # Call the original helper.  We leave ``latents``/``all_latents`` as None
    # for speed (i.e. we skip the DDIM inversion step used during evaluation).
    
    latents, all_latents = ie.invert(img_pil, canny, num_inference_steps, controlnet_conditioning_scale)


    result = ie.inference(
        source_prompt=source_prompt.strip(),
        control_image=control_image,
        target_prompt=edit_prompt.strip(),
        positive_prompt=positive_prompt.strip(),
        negative_prompt=negative_prompt.strip(),
        guidance_dpg=dpg_weight,
        guidance_cfg=cfg_weight,
        num_inference_steps=num_inference_steps,
        img=img_pil,
        thresh_e=mask_threshold,
        seed=seed,
        latents=latents,
        all_latents=all_latents,
        local=local,
    )

    return result


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="InstantEdit Demo") as demo:
        gr.Markdown(
            """
            # InstantEdit Interactive Demo
            Upload an image, write the *source prompt* that describes the **original**
            scene, then an *edit prompt* describing the **desired** change.  Adjust
            the sliders if you want to experiment with the guidance weights or the
            number of denoising steps, and hit **Run**.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(type="pil", label="Input image", height=320)
                source_prompt = gr.Textbox(label="Source prompt", lines=2)
                edit_prompt = gr.Textbox(label="Edit prompt", lines=2)
                positive_prompt = gr.Textbox(label="Positive prompt (optional)", value="", lines=1)
                negative_prompt = gr.Textbox(label="Negative prompt (optional)", value="", lines=1)
                local = gr.Textbox(label="Local blend word (optional)", value="", lines=1)

            with gr.Column(scale=1):
                num_steps = gr.Slider(1, 50, value=4, step=1, label="Inference steps")
                mask_thresh = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Mask threshold")
                cond_scale = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="ControlNet conditioning scale")
                dpg = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="DPG guidance weight")
                cfg = gr.Slider(0.0, 20.0, value=1.1, step=0.1, label="CFG guidance weight")
                seed = gr.Number(value=0, label="Seed", precision=0)

        out_image = gr.Image(type="pil", label="Edited image", height=512)

        run_btn = gr.Button("Run")
        run_btn.click(
            fn=generate,
            inputs=[
                img_input,
                source_prompt,
                edit_prompt,
                positive_prompt,
                negative_prompt,
                num_steps,
                mask_thresh,
                cond_scale,
                dpg,
                cfg,
                seed,
                local,
            ],
            outputs=out_image,
            api_name="generate",
        )

    return demo


def main():
    demo = build_demo()
    demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":
    main()
