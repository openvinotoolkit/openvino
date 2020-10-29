import java.util.HashMap;
import java.util.Map;

public class ArgumentParser {
    private Map<String, String> input;
    private Map<String, String> description;
    private String help;

    ArgumentParser(String help) {
        input = new HashMap<>();
        description = new HashMap<>();
        this.help = help;
    }

    public void addArgument(String arg, String help) {
        description.put(arg, help);
    }

    private void printHelp() {
        System.out.println(help);
        System.out.println('\n' + "Options:");
        for (Map.Entry<String, String> entry : description.entrySet()) {
            System.out.println("    " + entry.getKey() + "      " + entry.getValue());
        }
    }

    public void parseArgs(String[] args) {
        try {
            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                if (arg.equals("--help") | arg.equals("-h")) {
                    printHelp();
                    System.exit(0);
                } else {
                    if (description.containsKey(arg)) {
                        input.put(arg, args[++i]);
                    } else {
                        System.out.println("Non-existent key: '" + arg + "'");
                        System.exit(0);
                    }
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            System.out.println("Error: Incorrect number of arguments");
            System.exit(0);
        }
    }

    private String get(String flag) {
        return input.get(flag);
    }

    public String get(String flag, String defaultValue) {
        String res = input.get(flag);
        return (res != null) ? res : defaultValue;
    }

    public int getInteger(String flag, int defaultValue) {
        String res = get(flag);
        return (res != null) ? Integer.parseInt(res) : defaultValue;
    }

    public boolean getBoolean(String flag, boolean defaultValue) {
        String res = get(flag);
        return (res != null) ? Boolean.parseBoolean(res) : defaultValue;
    }
}
