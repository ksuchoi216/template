// DEFAULT

// SELECTOR
signx = document.getElementById("signX");
signo = document.getElementById("signO");
console.log(signx);
// EVENT
signx.addEventListener("click", gameStart(e));
signo.addEventListener("click", gameStart(e));

function gameStart(e) {
    console.log(e);
}


// SELECT MARER
// PLAY ROUND OF GAME
// - CHOOSE 1 CELL
// - AI CHOOSE 1 CELL
// - CHECK 3 IN-A-ROW
// - REPEAT  