# ProbitOptimization

<h1>Probit optimization in Julia</h1>

This is a simple MLE of a Probit model using Julia. Its main purpuse is to run through the various optimizers in the NLopt and Optim packages. Though it is is a simple problem, it is instructive.

<h1>LaTeX, Julia and Unicode</h1>

There is an example writeup in LaTeX that will add in the Julia code using the Listings package. Since my Julia code includes Unicode characters (Greek letters like β), I have to use a system font (Consolas on Windows) that has these characters. The julia-listings.tex file has the basic listings style file for Julia code, as well as the tweaks required to make listings display the Unicode characters properly. I have added most of the Greek alphabet, but if you use other characters, they will need to be added.
