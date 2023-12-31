{
    description = "A very basic flake";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
        flake-utils.url = "github:numtide/flake-utils";
    };

    outputs = { self, nixpkgs, flake-utils }:
        flake-utils.lib.eachDefaultSystem ( system:
        let
            pkgs = nixpkgs.legacyPackages.${ system };
        in {
            devShell = with pkgs; pkgs.mkShell rec {
                buildInputs = [
                    # build
                    python312
                    poetry
                    zlib
                    google-cloud-sdk

                    # rust build for tiktoken
                    llvmPackages.llvm
                    llvmPackages.clang
                    rustc
                ];

                shellHook = ''
                    # numpy
                    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
                    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"

                    # source environment variables
                    set -a; source .env; set +a

                    # install python packages
                    #poetry install

                    # gcloud
                    gcloud config set project $POETRY_PROJECT_ID

                    DEFAULT_CREDENTIALS_FILE="$HOME/.config/gcloud/application_default_credentials.json"
                    if [ -f "$DEFAULT_CREDENTIALS_FILE" ]; then
                        echo "Default credentials found. You are likely authenticated."
                    else
                        echo "Default credentials not found. Attempting to authenticate..."
                        gcloud auth application-default login
                    fi
                '';
            };
        }
    );
}

