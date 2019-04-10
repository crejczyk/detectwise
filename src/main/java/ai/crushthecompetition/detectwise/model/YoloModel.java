package ai.crushthecompetition.detectwise.model;
import java.io.IOException;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.model.YOLO2;

public class YoloModel extends YoloBaseModel {

	private static final YoloModel INSTANCE = new YoloModel();

    public static YoloModel getINSTANCE() {
        return INSTANCE;
    }
    
    private YoloModel() {
        try {
            preTrained = (ComputationGraph) YOLO2.builder().build().initPretrained();
            prepareLabels();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

	public String getLabels() {
		return "person\n" + "bicycle\n" + "car\n" + "motorbike\n" + "aeroplane\n" + "bus\n" + "train\n" + "truck\n"
				+ "boat\n" + "traffic light\n" + "fire hydrant\n" + "stop sign\n" + "parking meter\n" + "bench\n"
				+ "bird\n" + "cat\n" + "dog\n" + "horse\n" + "sheep\n" + "cow\n" + "elephant\n" + "bear\n" + "zebra\n"
				+ "giraffe\n" + "backpack\n" + "umbrella\n" + "handbag\n" + "tie\n" + "suitcase\n" + "frisbee\n"
				+ "skis\n" + "snowboard\n" + "sports ball\n" + "kite\n" + "baseball bat\n" + "baseball glove\n"
				+ "skateboard\n" + "surfboard\n" + "tennis racket\n" + "bottle\n" + "wine glass\n" + "cup\n" + "fork\n"
				+ "knife\n" + "spoon\n" + "bowl\n" + "banana\n" + "apple\n" + "sandwich\n" + "orange\n" + "broccoli\n"
				+ "carrot\n" + "hot dog\n" + "pizza\n" + "donut\n" + "cake\n" + "chair\n" + "sofa\n" + "pottedplant\n"
				+ "bed\n" + "diningtable\n" + "toilet\n" + "tvmonitor\n" + "laptop\n" + "mouse\n" + "remote\n"
				+ "keyboard\n" + "cell phone\n" + "microwave\n" + "oven\n" + "toaster\n" + "sink\n" + "refrigerator\n"
				+ "book\n" + "clock\n" + "vase\n" + "scissors\n" + "teddy bear\n" + "hair drier\n" + "toothbrush\n";

	}

}