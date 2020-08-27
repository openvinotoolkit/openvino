import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.junit.runner.notification.Failure;

public class OpenVinoTestRunner {

    public static void main(String[] args) {
        ArgumentParser parser = new ArgumentParser("");
        parser.addArgument("-d", "device to test");
        parser.parseArgs(args);
        IETest.device = parser.get("-d", "CPU");

        Result result = JUnitCore.runClasses(TestsSuite.class);
		
        for (Failure failure : result.getFailures()) {
           System.out.println(failure.toString());
        }
    }
}
