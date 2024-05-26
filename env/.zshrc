autoload -Uz promptinit
promptinit
prompt adam1

setopt promptsubst          # enable prompt substitution in prompt
setopt interactivecomments  # allow comments in interactive mode
setopt histignorealldups sharehistory

# Use emacs keybindings even if our EDITOR is set to vi
bindkey -e

# Keep 1000 lines of history within the shell and save it to ~/.zsh_history:
HISTSIZE=1000
SAVEHIST=1000
HISTFILE=~/.zsh_history

eval "$(dircolors -b)"

# Use modern completion system
autoload -Uz compinit
compinit

# do menu-driven selection
zstyle ':completion:*' menu select
# make it case-insensitive for completion
zstyle ':completion:*' matcher-list '' 'm:{a-z}={A-Z}' 'm:{a-zA-Z}={A-Za-z}' 'r:|[._-]=* r:|=* l:|=*'
# add ls colors to completion candidates
zstyle ':completion:*' list-colors ${(s.:.)LS_COLORS}

zstyle ':completion:*:*:kill:*:processes' list-colors '=(#b) #([0-9]#)*=0=01;31'
zstyle ':completion:*:kill:*' command 'ps -u $USER -o pid,%cpu,tty,cputime,cmd'

bindkey "^[[1;5C" forward-word      # ctrl + right
bindkey "^[[1;5D" backward-word     # ctrl + left
bindkey "^[[H" beginning-of-line    # home
bindkey "^[[F" end-of-line          # end
bindkey "^[[3~" delete-char         # delete
bindkey \^U backward-kill-line      # ctrl + u
bindkey \^H backward-kill-word      # ctrl + backspace (ctrl + h)
bindkey "^[[3;5~" kill-word         # ctrl + delete

alias ls='ls --color=auto'
alias ll='ls -lah'
alias grep='grep --color=auto'

export PATH="/usr/local/cuda/bin:${PATH}"
