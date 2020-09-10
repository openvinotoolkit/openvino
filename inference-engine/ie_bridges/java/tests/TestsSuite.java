import junit.framework.TestSuite;

import org.intel.openvino.*;
import org.junit.runner.RunWith;
import org.junit.runners.AllTests;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.*;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.*;

@RunWith(AllTests.class)
public class TestsSuite extends IETest {

    public static TestSuite suite() {
        TestSuite suite = new TestSuite();
        try {
            // get openvino_test.jar path
            String dir =
                    new File(
                                    TestsSuite.class
                                            .getProtectionDomain()
                                            .getCodeSource()
                                            .getLocation()
                                            .toURI())
                            .getPath()
                            .toString();

            List<Class<?>> results = findClasses(dir);
            for (Class<?> cl : results) {
                if (cl.getName() == "ArgumentParser") continue;
                suite.addTest(new junit.framework.JUnit4TestAdapter(cl));
            }
        } catch (ClassNotFoundException e) {
            System.out.println("ClassNotFoundException: " + e.getMessage());
        } catch (URISyntaxException e) {
            System.out.println("URISyntaxException: " + e.getMessage());
        }
        return suite;
    }

    private static List<Class<?>> findClasses(String directory) throws ClassNotFoundException {
        List<Class<?>> classes = new ArrayList<Class<?>>();
        try {
            ZipInputStream zip = new ZipInputStream(new FileInputStream(directory));
            for (ZipEntry entry = zip.getNextEntry(); entry != null; entry = zip.getNextEntry()) {
                String name = entry.getName().toString();
                if (name.endsWith(".class")
                        && !name.contains("$")
                        && !name.contains("/")
                        && !name.equals("TestsSuite.class")
                        && !name.equals("OpenVinoTestRunner.class")
                        && !name.equals("IETest.class")) {
                    classes.add(
                            Class.forName(name.substring(0, name.length() - ".class".length())));
                }
            }
        } catch (FileNotFoundException e) {
            System.out.println("FileNotFoundException: " + e.getMessage());
        } catch (IOException e) {
            System.out.println("IOException: " + e.getMessage());
        }
        return classes;
    }
}
