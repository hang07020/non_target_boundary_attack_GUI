from __future__ import print_function
import numpy as np
from PIL import Image
import streamlit as st

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

@st.cache_resource
def load_model():
    return ResNet50(weights='imagenet')

def orthogonal_perturbation(delta, prev_sample, target_sample):
    prev_sample = prev_sample.reshape(224, 224, 3)
    perturb = np.random.randn(224, 224, 3)
    perturb /= get_diff(perturb, np.zeros_like(perturb))
    perturb *= delta * np.mean(get_diff(target_sample, prev_sample))
    diff = (target_sample - prev_sample).astype(np.float32)
    diff /= get_diff(target_sample, prev_sample)
    diff = diff.reshape(3, 224, 224)
    perturb = perturb.reshape(3, 224, 224)
    for i, channel in enumerate(diff):
        perturb[i] -= np.dot(perturb[i], channel) * channel
    mean = [103.939, 116.779, 123.68]
    perturb = perturb.reshape(224, 224, 3)
    overflow = (prev_sample + perturb) - np.concatenate((np.ones((224, 224, 1)) * (255. - mean[0]), np.ones((224, 224, 1)) * (255. - mean[1]), np.ones((224, 224, 1)) * (255. - mean[2])), axis=2)
    perturb -= overflow * (overflow > 0)
    underflow = np.concatenate((np.ones((224, 224, 1)) * (0. - mean[0]), np.ones((224, 224, 1)) * (0. - mean[1]), np.ones((224, 224, 1)) * (0. - mean[2])), axis=2) - (prev_sample + perturb)
    perturb += underflow * (underflow > 0)
    return perturb

def forward_perturbation(epsilon, prev_sample, target_sample):
    perturb = (target_sample - prev_sample).astype(np.float32)
    perturb /= get_diff(target_sample, prev_sample)
    perturb *= epsilon
    return perturb

def postprocess_for_display(sample):
    sample = sample.reshape(224, 224, 3)
    mean = [103.939, 116.779, 123.68]
    sample = sample.copy()
    sample[..., 0] += mean[0]
    sample[..., 1] += mean[1]
    sample[..., 2] += mean[2]
    sample = sample[..., ::-1]
    sample = np.clip(sample, 0, 255).astype(np.uint8)
    return Image.fromarray(sample)

def preprocess(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def get_diff(sample_1, sample_2):
    sample_1 = sample_1.reshape(3, 224, 224)
    sample_2 = sample_2.reshape(3, 224, 224)
    diff = []
    for i, channel in enumerate(sample_1):
        diff.append(np.linalg.norm((channel - sample_2[i]).astype(np.float32)))
    return np.array(diff)

def get_sc_img():
    red, green, blue = 0, 0, 255
    height, width = 224, 224
    blue_image = Image.new('RGB', (width, height), (red, green, blue))
    return blue_image

def run_blue_attack(original_image_pil, max_steps, progress_bar, image_placeholder):
    classifier = load_model()
    
    initial_sample = preprocess(original_image_pil)
    target_sample = preprocess(get_sc_img())
    
    attack_class = np.argmax(classifier.predict(initial_sample))
    adversarial_sample = initial_sample
    
    epsilon = 1.
    delta = 0.1

    # Move first step to the boundary
    while True:
        trial_sample = adversarial_sample + forward_perturbation(epsilon * get_diff(adversarial_sample, target_sample), adversarial_sample, target_sample)
        prediction = classifier.predict(trial_sample.reshape(1, 224, 224, 3))
        if np.argmax(prediction) == attack_class:
            epsilon /= 0.9
        else:
            adversarial_sample = trial_sample
            break
        if epsilon < 1e-6:
             st.error("Could not find the decision boundary.")
             return postprocess_for_display(adversarial_sample)

    target_class = np.argmax(classifier.predict(adversarial_sample))

    # Main attack loop
    for n_steps in range(max_steps):

        # Delta step
        for _ in range(10):
            trial_samples = [adversarial_sample + orthogonal_perturbation(delta, adversarial_sample, initial_sample) for _ in range(10)]
            predictions = np.argmax(classifier.predict(np.array(trial_samples).reshape(-1, 224, 224, 3)), axis=1)
            d_score = np.mean(predictions == target_class)
            
            if d_score > 0.0:
                if d_score < 0.3: delta *= 0.9
                elif d_score > 0.7: delta /= 0.9
                adversarial_sample = np.array(trial_samples)[np.where(predictions == target_class)[0][0]]
                break 
            else:
                delta *= 0.9

        # Epsilon step
        for _ in range(10):
            trial_sample = adversarial_sample + forward_perturbation(epsilon * get_diff(adversarial_sample, initial_sample), adversarial_sample, initial_sample)
            prediction = classifier.predict(trial_sample.reshape(1, 224, 224, 3))
            if np.argmax(prediction) == target_class:
                adversarial_sample = trial_sample
                epsilon /= 0.5
                break
            else:
                epsilon *= 0.5

        # Update GUI
        progress_bar.progress((n_steps + 1) / max_steps, text=f"Step {n_steps + 1}/{max_steps}")
        if (n_steps + 1) % 10 == 0 or n_steps == max_steps - 1:
            image_placeholder.image(postprocess_for_display(adversarial_sample), caption=f"Adversarial Image (Step {n_steps + 1})", use_container_width=True)
    
    progress_bar.empty()
    return postprocess_for_display(adversarial_sample)
