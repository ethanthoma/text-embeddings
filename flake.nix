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
                    python39
                    poetry
                    zlib
                    google-cloud-sdk
                ];

                LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];

                shellHook = ''
                    mkdir -p $PWD/.local
                    export HOME=$PWD/.home
                    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
                    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
                    poetry install

                    set -a; source .env; set +a

                    gcloud config set project $POETRY_PROJECT_ID
                    gcloud auth application-default login
                '';
            };
        }
    );
}

