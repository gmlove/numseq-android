package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;

import java.io.IOException;
import java.util.*;

public class BboxImageDetector implements Classifier {
    static {
        System.loadLibrary("tensorflow_numseq");
    }

    private static final String TAG = "BboxImageDetector";

    // Config values.
    public static final String INPUT_NAME = "input-bbox:0";
    public static final String OUTPUT_NAME = "output-bbox:0";
    public static final String INITIALIZER_NAME = "initializer-bbox";
    public static final int INPUT_SIZE = 64;
    public static final int IMAGE_MEAN = 117;
    public static final float IMAGE_STD = 1;
    public static final String MODEL_FILE = "file:///android_asset/graph-bbox.pb";

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    private float[] floatValues;
    private float[] outputs;
    private String[] outputNames;

    private TensorFlowInferenceInterface inferenceInterface;

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     * @return The native return value, 0 indicating success.
     * @throws IOException
     */
    public void initializeTensorFlow(
            AssetManager assetManager) throws IOException {

        // Pre-allocate buffers.
        outputNames = new String[] {OUTPUT_NAME};
        intValues = new int[INPUT_SIZE * INPUT_SIZE];
        floatValues = new float[INPUT_SIZE * INPUT_SIZE * 3];
        outputs = new float[4];

        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.enableStatLogging(true);

        if (inferenceInterface.initializeTensorFlow(assetManager, MODEL_FILE) != 0) {
            throw new RuntimeException("TF initialization failed");
        }

        if (inferenceInterface.initializeModel(INITIALIZER_NAME) != 0) {
            throw new RuntimeException("TF initialization failed");
        }
    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
            floatValues[i * 3 + 2] = ((val & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
        }
        Trace.endSection();

        // Copy the input data into TensorFlow.
        Trace.beginSection("fillNodeFloat");
        inferenceInterface.fillNodeFloat(
                INPUT_NAME, new int[] {1, INPUT_SIZE, INPUT_SIZE, 3}, floatValues);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("runInference");
        inferenceInterface.runInference(outputNames);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("readNodeFloat");
        inferenceInterface.readNodeFloat(OUTPUT_NAME, outputs);
        Trace.endSection();

        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        recognitions.add(getRecognitions(outputs));

        Trace.endSection(); // "recognizeImage"
        return recognitions;
    }

    private Recognition getRecognitions(float[] outputs) {
        String title = String.format("bbox[%s]: [%.2f, %.2f, %.2f, %.2f]",
                outputs.length, outputs[0], outputs[1], outputs[2], outputs[3]);
        RectF bbox = new RectF(outputs[0], outputs[1], outputs[0] + outputs[2], outputs[1] + outputs[3]);
        return new Recognition("0", title, 1.0f, bbox);
    }

    public void enableStatLogging(boolean debug) {
        inferenceInterface.enableStatLogging(debug);
    }

    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }
}
