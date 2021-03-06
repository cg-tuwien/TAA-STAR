# TAA-STAR

Implementations of several state of the art Temporal Anti-Aliasing (TAA) techniques.

## Project Description

TBD.

## Setup Instructions and Submodules

1. Clone the repository with all submodules recursively:       
`git clone https://github.com/cg-tuwien/TAA-STAR.git . --recursive` (to check out into `.`)
2. Open `taa.sln` with Visual Studio 2019, set `taa` as the startup project, build and run

You might want to have all submodules checked-out to their `master` branch. You can do so using:      
`git submodule foreach --recursive git checkout master`.       
There are two submodules: One under `gears_vk/` (referencing https://github.com/cg-tuwien/Gears-Vk) and another under `gears_vk/auto_vk/` (referencing https://github.com/cg-tuwien/Auto-Vk).    

To update the submodules on a daily basis, use the following command:  
`git submodule foreach --recursive 'git checkout master && git pull'`

To contribute to either of the submodules, please do so via pull requests and follow the ["Contributing Guidelines" from Gears-Vk](https://github.com/cg-tuwien/Gears-Vk/blob/master/CONTRIBUTING.md). Every time you check something in, make sure that the correct submodule-commits (may also reference forks) are referenced so that one can always get a compiling and working version by cloning as described in step 1!

## Scene Setup

1. Download the Emerald Square scene from https://developer.nvidia.com/orca/nvidia-emerald-square
2. Extract it to a (new) folder of your choice
3. Copy the file `extras/EmeraldSquare_Day.fscene` from the repository into that folder
4. Launch the program: `taa.exe <full path to the .fscene file in your Emerald-Square-folder>`

If no scene file is specified, the included Sponza-scene is used.
There is also a very simple, fast-loading test scene included in `extras/TestScene/`.

## Documentation 

TBD.

## License 

TBD.
