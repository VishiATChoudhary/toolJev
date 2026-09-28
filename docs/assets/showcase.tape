# Records docs/assets/showcase.gif. Run from the repo root:  vhs docs/assets/showcase.tape
Output docs/assets/showcase.gif

Set Shell zsh
Set FontSize 17
Set Width 1400
Set Height 900
Set Padding 28
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms
Set WindowBar Colorful

Hide
Type "export PS1='$ ' && uv run python -c 1 && clear"
Enter
Sleep 500ms
Show

Type "uv run python examples/showcase.py"
Sleep 500ms
Enter
Wait+Screen@90s /one execute call/
Sleep 5s
