package ai.crushthecompetition.detectwise;
import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_DUPLEX;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TinyYoloModel {

	private static final Logger LOGGER = LoggerFactory.getLogger(TinyYoloModel.class);
	
	private static final TinyYoloModel INSTANCE = new TinyYoloModel();

    private ComputationGraph preTrained;
    private List<DetectedObject> predictedObjects;
    private HashMap<Integer, String> labels;

    public static TinyYoloModel getINSTANCE() {
        return INSTANCE;
    }
    
    private TinyYoloModel() {
        try {
            preTrained = (ComputationGraph) YOLO2.builder().build().initPretrained();
            prepareLabels();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void markObjWithBoundingBox(Mat file, int imageWidth, int imageHeight, boolean newBoundingBOx,String winName) throws Exception {
        int width = 416; // Width of the video frame 
        int height = 416; // Height of the video frame
        int gridWidth = 13; // Grid width
        int gridHeight = 13; // Grid Height
        double detectionThreshold = 0.6; // Detection threshold

        Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) preTrained.getOutputLayer(0);
        if (newBoundingBOx) {
            INDArray indArray = prepareImage(file, width, height);
            INDArray results = preTrained.outputSingle(indArray);
            predictedObjects = outputLayer.getPredictedObjects(results, detectionThreshold);
            LOGGER.info(String.format("results = " + predictedObjects));
            markObjWithBoundingBox(file, gridWidth, gridHeight, imageWidth, imageHeight);
        } else {
        	markObjWithBoundingBox(file, gridWidth, gridHeight, imageWidth, imageHeight);
        }
        imshow(winName, file);
    }

    private INDArray prepareImage(Mat file, int width, int height) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(height, width, 3);
        ImagePreProcessingScaler imagePreProcessingScaler = new ImagePreProcessingScaler(0, 1);
        INDArray indArray = loader.asMatrix(file);
        imagePreProcessingScaler.transform(indArray);
        return indArray;
    }

	private void prepareLabels() {
		if (labels == null) {
			String s = "person\n" + "bicycle\n" + "car\n" + "motorbike\n" + "aeroplane\n" + "bus\n" + "train\n"
					+ "truck\n" + "boat\n" + "traffic light\n" + "fire hydrant\n" + "stop sign\n" + "parking meter\n"
					+ "bench\n" + "bird\n" + "cat\n" + "dog\n" + "horse\n" + "sheep\n" + "cow\n" + "elephant\n"
					+ "bear\n" + "zebra\n" + "giraffe\n" + "backpack\n" + "umbrella\n" + "handbag\n" + "tie\n"
					+ "suitcase\n" + "frisbee\n" + "skis\n" + "snowboard\n" + "sports ball\n" + "kite\n"
					+ "baseball bat\n" + "baseball glove\n" + "skateboard\n" + "surfboard\n" + "tennis racket\n"
					+ "bottle\n" + "wine glass\n" + "cup\n" + "fork\n" + "knife\n" + "spoon\n" + "bowl\n" + "banana\n"
					+ "apple\n" + "sandwich\n" + "orange\n" + "broccoli\n" + "carrot\n" + "hot dog\n" + "pizza\n"
					+ "donut\n" + "cake\n" + "chair\n" + "sofa\n" + "pottedplant\n" + "bed\n" + "diningtable\n"
					+ "toilet\n" + "tvmonitor\n" + "laptop\n" + "mouse\n" + "remote\n" + "keyboard\n" + "cell phone\n"
					+ "microwave\n" + "oven\n" + "toaster\n" + "sink\n" + "refrigerator\n" + "book\n" + "clock\n"
					+ "vase\n" + "scissors\n" + "teddy bear\n" + "hair drier\n" + "toothbrush\n";
			
			String[] split = s.split("\\n");
			int i = 0;
			labels = new HashMap<>();
			for (String s1 : split) {
				labels.put(i++, s1);
			}
		}
	}

    private void markObjWithBoundingBox(Mat file, int gridWidth, int gridHeight, int w, int h) {
        if (predictedObjects == null) {
            return;
        }
        ArrayList<DetectedObject> detectedObjects = new ArrayList<>(predictedObjects);

        while (!detectedObjects.isEmpty()) {
            Optional<DetectedObject> max = detectedObjects.stream().max((o1, o2) -> ((Double) o1.getConfidence()).compareTo(o2.getConfidence()));
            if (max.isPresent()) {
                DetectedObject maxObjectDetect = max.get();
                removeObjectsIntersectingWithMax(detectedObjects, maxObjectDetect);
                detectedObjects.remove(maxObjectDetect);
                markObjWithBoundingBox(file, gridWidth, gridHeight, w, h, maxObjectDetect);
            }
        }
    }

	private void removeObjectsIntersectingWithMax(ArrayList<DetectedObject> detectedObjects,
			DetectedObject maxObjectDetect) {
		double[] bottomRightXY1 = maxObjectDetect.getBottomRightXY();
		double[] topLeftXY1 = maxObjectDetect.getTopLeftXY();
		List<DetectedObject> removeIntersectingObjects = new ArrayList<>();
		for (DetectedObject detectedObject : detectedObjects) {
			double[] topLeftXY = detectedObject.getTopLeftXY();
			double[] bottomRightXY = detectedObject.getBottomRightXY();
			double iox1 = Math.max(topLeftXY[0], topLeftXY1[0]);
			double ioy1 = Math.max(topLeftXY[1], topLeftXY1[1]);

			double iox2 = Math.min(bottomRightXY[0], bottomRightXY1[0]);
			double ioy2 = Math.min(bottomRightXY[1], bottomRightXY1[1]);

			double inter_area = (ioy2 - ioy1) * (iox2 - iox1);

			double box1_area = (bottomRightXY1[1] - topLeftXY1[1]) * (bottomRightXY1[0] - topLeftXY1[0]);
			double box2_area = (bottomRightXY[1] - topLeftXY[1]) * (bottomRightXY[0] - topLeftXY[0]);

			double union_area = box1_area + box2_area - inter_area;
			double iou = inter_area / union_area;
			if (iou > 0.5) {
				removeIntersectingObjects.add(detectedObject);
			}
		}
		detectedObjects.removeAll(removeIntersectingObjects);
	}

    private void markObjWithBoundingBox(Mat file, int gridWidth, int gridHeight, int w, int h, DetectedObject obj) {
        double[] xy1 = obj.getTopLeftXY();
        double[] xy2 = obj.getBottomRightXY();
        int predictedClass = obj.getPredictedClass();
        int x1 = (int) Math.round(w * xy1[0] / gridWidth);
        int y1 = (int) Math.round(h * xy1[1] / gridHeight);
        int x2 = (int) Math.round(w * xy2[0] / gridWidth);
        int y2 = (int) Math.round(h * xy2[1] / gridHeight);
        rectangle(file, new Point(x1, y1), new Point(x2, y2), Scalar.BLUE);
        putText(file, labels.get(predictedClass), new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.MAGENTA);
        putText(file, String.valueOf(obj.getConfidence()), new Point(x1 + 2, y2 + 20), FONT_HERSHEY_DUPLEX, 0.5, Scalar.WHITE);
    }

}