{ pkgs ? import <nixpkgs> {} }:

let
  # Override to enable tests if needed (default: false for reliable builds)
  enableTests = false;
  
  # Build OpenVINO with the specified test configuration
  openvino = pkgs.callPackage ./default.nix {
    inherit enableTests;
    python3 = pkgs.python3;
    cmake = pkgs.cmake;
    ninja = pkgs.ninja;
  };
  
  # Runtime packages that might be needed
  runtimePackages = [ 
    pkgs.pkg-config 
    pkgs.level-zero 
    pkgs.opencv
    pkgs.protobuf
    pkgs.flatbuffers
    pkgs.pugixml
    pkgs.snappy
    pkgs.tbb
    pkgs.gflags
    pkgs.libusb1
    pkgs.libxml2
    pkgs.ocl-icd
  ];
  
in pkgs.mkShell rec {
  name = "openvino-dev";

  buildInputs = runtimePackages;

  hardeningDisable = [ "zerocallusedregs" ];

  shellHook = ''
    echo "🚀 OpenVINO Development Environment"
    echo "📦 Test building: ${if enableTests then "ENABLED" else "DISABLED"}"
    echo "🔧 To enable tests, set enableTests = true in shell.nix"
    echo ""
    echo "Available commands:"
    echo "  nix-build -A openvino    # Build OpenVINO"
    echo "  nix-shell                 # Enter this environment"
    echo ""
    
    export ZDOTDIR="$(mktemp -d)"
    cat > "$ZDOTDIR/.zshrc" << 'EOF'
      source ~/.zshrc

      function parse_git_branch {
        git branch --no-color 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/\ ->\ \1/'
      }

      function display_jobs_count_if_needed {
        local job_count=$(jobs -s | wc -l | tr -d " ")

        if [ $job_count -gt 0 ]; then
          echo "%B%F{yellow}%j| ";
        fi
      }

      PROMPT="%F{blue}$(date +%H:%M:%S) $(display_jobs_count_if_needed)%B%F{green}%n %F{blue}%~%F{cyan} ❄%F{yellow}$(parse_git_branch) %f%{$reset_color%}"
    EOF

    if [ -z "$DIRENV_IN_ENVRC" ]; then
      exec zsh -i
    fi
  '';
}
