## CS workflow with template
```mermaid
sequenceDiagram
    participant User
    participant Template
    participant Config Manager
    participant Mode
    participant Traversal
    participant Commit Path
    User->>Template: Runs commit_slider.py
    Note right of User: Passes parameters via config or CLI
    Template->>Config Manager: Validates parameters and generates corresponding config
    Note right of Template: Determines Mode and Traversal
    Config Manager->>Mode: Prepares config. (normalization)
    Note right of Config Manager: Makes paths absolutize, checks config. consistency
    loop MainLoop
        Traversal->>Mode: Slides to another commit
        Mode->>Traversal: Resolves current commit
        Traversal->>Commit Path: Populates by current commit
    end
    Note right of Commit Path: Mode appends additional info,<br>corresponding commit state and target task
    Commit Path->>Mode: Commit path with details about each passed commit
    Template-->>Mode: Representative function
    Mode->>User: Formatted output
```

## CS workflow without template (pure configuration)
```mermaid
sequenceDiagram
participant User
participant Config Manager
participant Mode
participant Traversal
participant Commit Path
User->>Config Manager: Runs commit_slider.py with configuration
Note right of User: Validates and prepares configuration,<br>Mode and Traversal are determined by User directly
Config Manager->>Mode: Prepares config. (normalization)
Note right of Config Manager: Makes paths absolutize, checks config. consistency
loop MainLoop
Traversal->>Mode: Slides to another commit
Mode->>Traversal: Resolves current commit
Traversal->>Commit Path: Populates by current commit
end
Note right of Commit Path: Mode appends additional info,<br>corresponding commit state and target task
Commit Path->>Mode: Commit path with details about each passed commit
Mode->>User: Formatted output
```
