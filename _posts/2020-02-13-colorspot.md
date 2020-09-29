---
title: "Colorspot"
date: 2020-02-13
tags: [unity, games, game development, C#]
header:
  image: "/images/data_art.png"
excerpt: "A simple catch and fall game developed with Unity Engine & C#"
classes: wide
---

<style>
b {
    color: #f25278;
}

i {
    color: #f25278;
}

body {
    text-align: justify;
    font-size: 15px;
}
</style>

## A simple catch and fall game developed with Unity Engine & C#

<b>Tools Used</b>
<ul>
    <li>Unity</li>
    <li>C#</li>
    <li>Adobe Illustrator</li>
</ul>

---

Being a guy who likes to play games mainly Battle Royale (BRs) & First Person Shooter (FPS) games, 
I was always curious about how Computer/Console games are made. 
Hence, I decided to learn Game Development in Spring 2019 to get familiar with the field of Game Development. 
After a week of research about various development environments available for 
games and programming languages used, my search narrowed down to two engines: 
Unity and Unreal Engine. 
Since Unity uses Javascript or C# for development, I decided to use Unity with C# for the Project. 
I worked in a team of 6 for this Project (More details about team members available on Github repo page). We decided to build an educational and fun to play, catch and fall video game and named it Colorspot. To make it more interesting and educational, we added different color pallettes for each level.  
My team and I decided to go with the Agile software development lifecycle and used Trello board to track use cases.

The development stage was broken down into two parts: Menu User Interface (UI) and Gameplay.

---

<b> User Interface</b>

The User Interface was split into five different scenes: 
<ul>
    <li>
        <b>Load User</b>: The Load User scene is used to select one of the four users. If a user does not exist denoted by < None >, selecting the profile button prompts to enter a name for the user. To limit the number of userdata saved on disk, we limited number of users to 4. To streamline the process of user saving and loading from disk, a <i>User</i> class is used. When the user selects one of the profiles, the <code>makecurrentuser</code> function in User class loads the userData from disk into memory.
    </li>
    <li>
        <b>Level Select</b>: The Level Select screen contains options for <i>Quit</i>, <i>Previous level</i>, <i>Next level</i>, <i>Home</i> and <i>Level Selectors</i>. When a user clicks on a Level Selector, it changes the current scene to the game level selected out of Level 1, 2 or 3 scenes.
        <!-- add youtube video snippet here showing the select level scene, same for other scenes--> 
    </li>
    <li>
        <b>Menu</b>: The Menu button contains options for <i>Change User</i>, <i>Start game</i>, <i>Store</i>, <i>Settings</i> and <i>Exit</i>. Selecting the Settings option changes current scene to Settings, the Change User option is mapped to the Load User scene, the Start button option to the Level Select scene, the Store button is mapped to Shop scene and the Exit button closes the game.
    </li>
    <li>
        <b>Settings</b>: The Settings scene contains inputs for volume control which can be changed using the slider available in the scene.
    </li>
    <li>
        <b>Shop</b>: The Shop Scene contains two panels; one on the right and one at the bottom to display the available items to purchase. The selection panel on the right contains Sprite (or User controlled object), Note (or Falling object) and Background image. The bottom panel shows various available options to purchase for the selected object type in the right panel. The Select button at the bottom of the scene can be used to select the current object for gameplay. The user's current selection is saved to disk to remember the next time user logs into the game.
    </li>
</ul>

<b>Gameplay</b>

The Gameplay is divided into three scenes for three levels available:
<ul>
    <li>Level 1</li>
    <li>Level 2</li>
    <li>Level 3</li>
</ul>
Each level can be played for 60 seconds with a multiplier which increments after a caught node and resets on a missed note. The top panel on the level scene shows timer on the left, color pallete in the center and pause option on the right. The bottom panel shows the sprite selected, colors available in the pallette and the multiplier in order respectively. The level color pallette is applied to the falling notes. Level 1 contains primary colors, level 2 contains secondary colors and level 3 contains tertiary colors.This feature was added to make the game educational. The pause button pauses the timer and shows options to <i>Resume game</i>, <i>Quit game</i>, <i>Home</i> and <i>Restart</i>. The game ends when the timer reaches 0 after which a Final score is shown to the user. If the score exceeds 150, the players gets bonus points added to encourage user to catch as many notes as possible. If the user scores a high score, it is added to local disk for persistance. The user also earns coins equal to half the score which can be used to purchase items in shop.

---

<b>Demo</b>

<iframe width="560" height="315" src="https://www.youtube.com/embed/CNAdpm5gDwc" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---

<b>Code Repository</b>

Click <a href="https://github.com/kasim95/Unity_Colorspot" target="_blank">here</a> to view the Github repo.

---
