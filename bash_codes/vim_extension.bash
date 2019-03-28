apt install git
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
mkdir -p $(jupyter --data-dir)/nbextensions
cd $(jupyter --data-dir)/nbextensions
git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding
jupyter nbextension enable vim_binding/vim_binding
sed -i "/.*Ctrl-C.*:/d" ~/.local/share/jupyter/nbextensions/vim_binding/vim_binding.js
sed -i "/.*'Shift-Esc'.*:/a\      \'Ctrl-C\': CodeMirror.prototype.leaveInsertMode," ~/.local/share/jupyter/nbextensions/vim_binding/vim_binding.js
