package ai.crushthecompetition.detectwise;

import static org.bytedeco.javacpp.opencv_highgui.destroyAllWindows;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_highgui.waitKey;

import java.util.concurrent.ThreadLocalRandom;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.crushthecompetition.detectwise.model.YoloBaseModel;

public class VideoDetection {

	private static final Logger LOGGER = LoggerFactory.getLogger(VideoDetection.class);
	
	private static final String DETECT_WISE = "ia.crushthecompetition.detectwise";
	
	private volatile Frame videoFrame;
	private volatile Mat[] v = new Mat[1];
	private Thread thread;
	private volatile boolean stop = false;
	private String windowName;

	public void startRealTimeVideoDetection(String videoFileName, YoloBaseModel yoloModel) throws java.lang.Exception {
		try (FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoFileName)) {
			windowName =  DETECT_WISE + ThreadLocalRandom.current().nextInt();
			grabber.start();
			while (!stop) {
				videoFrame = grabber.grab();
				if (videoFrame == null) {
					stop();
					break;
				}
				v[0] = new OpenCVFrameConverter.ToMat().convert(videoFrame);
				if (v[0] == null) {
					continue;
				}

				if (thread == null) {
					thread = new Thread(() -> {
						while (videoFrame != null && !stop) {
							try {
								if (v[0] != null) {
									yoloModel.markObjWithBoundingBox(v[0], videoFrame.imageWidth,
											videoFrame.imageHeight, true, windowName);
								}
							} catch (java.lang.Exception e) {
								LOGGER.error(e.getMessage(),e);
							}
						}
					});
					thread.setDaemon(true);		
					thread.start();
				};
				
				yoloModel.markObjWithBoundingBox(v[0], videoFrame.imageWidth,
						videoFrame.imageHeight, false, windowName);
				imshow(windowName, v[0]);

				char key = (char) waitKey(20);
				if (key == 27) {
					stop();
					break;
				}
				Thread.sleep(60);
			}
		}
	}

	public void stop() {
		if (!stop) {
			stop = true;
			destroyAllWindows();
		}
	}
}