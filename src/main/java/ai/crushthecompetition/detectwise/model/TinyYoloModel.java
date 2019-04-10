package ai.crushthecompetition.detectwise.model;
import java.io.IOException;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.model.TinyYOLO;

public class TinyYoloModel extends YoloBaseModel {

	private static final TinyYoloModel INSTANCE = new TinyYoloModel();

    public static TinyYoloModel getINSTANCE() {
        return INSTANCE;
    }
    
    private TinyYoloModel() {
        try {
            preTrained = (ComputationGraph) TinyYOLO.builder().build().initPretrained();
            prepareLabels();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

	@Override
	protected String getLabels() {
		return "aeroplane\n" + "bicycle\n" + "bird\n" + "boat\n" + "bottle\n" + "bus\n" + "car\n" + "cat\n" + "chair\n"
				+ "cow\n" + "diningtable\n" + "dog\n" + "horse\n" + "motorbike\n" + "person\n" + "pottedplant\n"
				+ "sheep\n" + "sofa\n" + "train\n" + "tvmonitor";
	}

}