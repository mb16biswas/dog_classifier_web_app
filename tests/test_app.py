import pytest
import numpy as np
from PIL import Image
from io import BytesIO
import streamlit as st
from app import process  # Import the process function from your Streamlit app


# Test the process function
def test_process_function():
    # Simulate a 224x224 image with 3 channels (RGB)
    fake_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Call the process function
    result = process(fake_image)

    # Ensure that the result is a string and contains 'it is'
    assert isinstance(result, str), "The result should be a string"
    assert "it is" in result, f"Expected result to contain 'it is', but got {result}"