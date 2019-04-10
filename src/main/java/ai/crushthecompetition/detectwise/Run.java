package ai.crushthecompetition.detectwise;

import java.util.concurrent.Executors;

import javax.swing.JFrame;

import ai.crushthecompetition.detectwise.model.YoloModel;
import ai.crushthecompetition.detectwise.ui.ProgressBar;
import ai.crushthecompetition.detectwise.ui.UI;

public class Run {
	
    private static JFrame mainFrame = new JFrame();
    public static void main(String[] args) throws Exception {

        ProgressBar progressBar = new ProgressBar(mainFrame, true);
        progressBar.showProgressBar("Loading model...");
        UI ui = new UI();
        Executors.newCachedThreadPool().submit(()->{
            try {
                YoloModel yoloModel = YoloModel.getINSTANCE();
                ui.initUI(yoloModel);
            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                progressBar.setVisible(false);
                mainFrame.dispose();
            }
        });
    }
}