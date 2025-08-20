{
  description = "OpenVINO™ Toolkit - Open source toolkit for optimizing and deploying AI inference";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        
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
        
        # Build the OpenVINO package
        openvino = pkgs.callPackage ./default.nix {
          # Override any specific dependencies if needed
          python3 = pkgs.python3;
          cmake = pkgs.cmake;
          ninja = pkgs.ninja;
          # Add other overrides as needed
        };
        
      in {
        devShells.default = pkgs.mkShell rec {
          name = "openvino-dev";

          buildInputs = runtimePackages;

          hardeningDisable = [ "zerocallusedregs" ];

          shellHook = ''
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
        };

        packages = {
          # Main OpenVINO package
          openvino = openvino;
          
          # Package with tests enabled (for advanced users)
          openvino-with-tests = pkgs.callPackage ./default.nix {
            python3 = pkgs.python3;
            cmake = pkgs.cmake;
            ninja = pkgs.ninja;
            # Override to enable tests
            enableTests = true;
          };
        };
        
        # Default package
        defaultPackage = openvino;
      }
    );
}
