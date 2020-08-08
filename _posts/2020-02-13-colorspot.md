---
title: "Colorspot"
date: 2020-02-13
tags: [unity, games, game development, C#]
header:
  image: "/images/data_art.png"
excerpt: "A simple catch and fall game developed with Unity Engine"
---

<style>
.linebreak {
    border: 1px;
    border-color: red;
}

code {
    color: #f25278;
}

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

### A simple catch and fall game developed with Unity Engine

---

Tools Used:
<ol>
    <li><b>Unity</b></li> Used as the game engine.
    <li><b>C#</b></li> Used to write scripts for components and event listeners in Unity.
    <li><b>Visual Studio Code</b></li> Used as the editor for C#.
    <li><b>Adobe Illustrator</b></li> Used to design the sprites, notes, backgrounds and other UI elements.
    <li><b>Github</b></li> Used for version control.
</ol>

---

<details>
<summary>Project Directory Structure</summary>

<div makrdown="1">

```bash
📦Assets
 ┣ 📂Fonts
 ┃ ┣ 📜Simply Rounded.ttf
 ┣ 📂Music
 ┃ ┣ 📜(Music Box Remix) Pikmin - Forest of Hope.mp3
 ┃ ┣ 📜SelectionMusic.mp3
 ┣ 📂Physics Material
 ┃ ┣ 📂Prefabs
 ┃ ┃ ┣ 📜amber.prefab
 ┃ ┃ ┣ 📜amberbag.prefab
 ┃ ┃ ┣ 📜Angry_Raabbit.prefab
 ┃ ┃ ┣ 📜blue.prefab
 ┃ ┃ ┣ 📜bluebag.prefab
 ┃ ┃ ┣ 📜Cat.prefab
 ┃ ┃ ┣ 📜green.prefab
 ┃ ┃ ┣ 📜greenbag.prefab
 ┃ ┃ ┣ 📜magenta.prefab
 ┃ ┃ ┣ 📜magentabag.prefab
 ┃ ┃ ┣ 📜New_Flame_Amber.prefab
 ┃ ┃ ┣ 📜New_Flame_Blue.prefab
 ┃ ┃ ┣ 📜New_Flame_Green.prefab
 ┃ ┃ ┣ 📜New_Flame_Orange.prefab
 ┃ ┃ ┣ 📜New_Flame_Purple.prefab
 ┃ ┃ ┣ 📜New_Flame_Red.prefab
 ┃ ┃ ┣ 📜New_Flame_Teal.prefab
 ┃ ┃ ┣ 📜New_Flame_Violet.prefab
 ┃ ┃ ┣ 📜New_Flame_Yellow.prefab
 ┃ ┃ ┣ 📜orange.prefab
 ┃ ┃ ┣ 📜orangebag.prefab
 ┃ ┃ ┣ 📜purple.prefab
 ┃ ┃ ┣ 📜purplebag.prefab
 ┃ ┃ ┣ 📜rat_ball_amber.prefab
 ┃ ┃ ┣ 📜Rat_Ball_Blue.prefab
 ┃ ┃ ┣ 📜Rat_Ball_Green.prefab
 ┃ ┃ ┣ 📜Rat_Ball_Orange.prefab
 ┃ ┃ ┣ 📜Rat_Ball_Purple.prefab
 ┃ ┃ ┣ 📜Rat_Ball_Red.prefab
 ┃ ┃ ┣ 📜Rat_Ball_Teal.prefab
 ┃ ┃ ┣ 📜Rat_Ball_Violet.prefab
 ┃ ┃ ┣ 📜Rat_Ball_Yellow.prefab
 ┃ ┃ ┣ 📜red.prefab
 ┃ ┃ ┣ 📜redbag.prefab
 ┃ ┃ ┣ 📜Sad_Girl.prefab
 ┃ ┃ ┣ 📜Sword_Guy.prefab
 ┃ ┃ ┣ 📜teal.prefab
 ┃ ┃ ┣ 📜tealbag.prefab
 ┃ ┃ ┣ 📜yellow.prefab
 ┃ ┗ ┗ 📜yellowbag.prefab
 ┣ 📂Scenes
 ┃ ┣ 📜Level_1.unity
 ┃ ┣ 📜Level_2.unity
 ┃ ┣ 📜Level_3.unity
 ┃ ┣ 📜Level_Select.unity
 ┃ ┣ 📜Load User.unity
 ┃ ┣ 📜Menu2.unity
 ┃ ┣ 📜Settings.unity
 ┃ ┗ 📜Shop.unity
 ┣ 📂Screens
 ┃ ┣ 📂AI
 ┃ ┃ ┣ 📜ColorSpot_Frame.ai
 ┃ ┃ ┣ 📜ColorSpot_LevelSelector.ai
 ┃ ┃ ┣ 📜ColorSpot_Shop.ai
 ┃ ┃ ┣ 📜ColorSpot_Stage.ai
 ┃ ┃ ┣ 📜ColorSpot_StageEnd.ai
 ┃ ┃ ┣ 📜ColorSpot_StagePause.ai
 ┃ ┃ ┣ 📜ColorSpot_StartScreen.ai
 ┃ ┃ ┣ 📜ColorSpot_UserSelector.ai
 ┃ ┃ ┗ 📜ColorSpot_UserSelector_V2.ai
 ┃ ┣ 📂PNG
 ┃ ┃ ┣ 📜ColorSpot_LevelSelector.png
 ┃ ┃ ┣ 📜ColorSpot_Shop.png
 ┃ ┃ ┣ 📜ColorSpot_Stage.png
 ┃ ┃ ┣ 📜ColorSpot_StageEnd.png
 ┃ ┃ ┣ 📜ColorSpot_StagePause.png
 ┃ ┗ ┗ 📜ColorSpot_StartScreen.png
 ┣ 📂Scripts
 ┃ ┣ 📜AvatarChoice.cs
 ┃ ┣ 📜ChooseBackgrounds.cs
 ┃ ┣ 📜delayAudio.cs
 ┃ ┣ 📜DestroyOnContactA.cs
 ┃ ┣ 📜DestroyOnContactB.cs
 ┃ ┣ 📜DestroyOnContactBucket.cs
 ┃ ┣ 📜DestroyOnContactC.cs
 ┃ ┣ 📜GameController.cs
 ┃ ┣ 📜GameController2.cs
 ┃ ┣ 📜GameController3.cs
 ┃ ┣ 📜GameOver.cs
 ┃ ┣ 📜gameover1.cs
 ┃ ┣ 📜GameOver2.cs
 ┃ ┣ 📜GameOverPanel.cs
 ┃ ┣ 📜game_over2.cs
 ┃ ┣ 📜game_over3.cs
 ┃ ┣ 📜main_menu.cs
 ┃ ┣ 📜MenuScript.cs
 ┃ ┣ 📜MoveAvatar.cs
 ┃ ┣ 📜PauseMenu.cs
 ┃ ┣ 📜pause_level1.cs
 ┃ ┣ 📜pause_level2.cs
 ┃ ┣ 📜pause_level3.cs
 ┃ ┣ 📜ScoreA.cs
 ┃ ┣ 📜ScoreB.cs
 ┃ ┣ 📜ScoreC.cs
 ┃ ┣ 📜SelectMenu.cs
 ┃ ┣ 📜SelectMenuScript.cs
 ┃ ┣ 📜settings.cs
 ┃ ┣ 📜shop_script.cs
 ┃ ┣ 📜StayInside.cs
 ┃ ┃ 📜user_selection.cs
 ┃ ┣ 📜DestroyOnContactD.cs
 ┃ ┣ 📜DestroyOnContactE.cs
 ┃ ┣ 📜DestroyOnContactF.cs
 ┃ ┣ 📜DestroyOnContactG.cs
 ┃ ┣ 📜ScoreD.cs
 ┃ ┣ 📜ScoreE.cs
 ┃ ┣ 📜ScoreF.cs
 ┗ ┗ 📜ScoreG.cs
```
</div>
</details>

---

Being a guy who likes to play games mainly Battle Royale (BRs) & First Person Shooter (FPS) games, I was always curious about how Computer/Console games are made. Also, being a Computer Science student meant that I could learn Game Development as my career choice. Hence, I decided to learn Game Development in Spring 2019 to get familiar with the field of Game Development. After a week of research about various development environments available for games and programming languages used, my search narrowed down to two engines: Unity and Unreal Engine. Since Unity uses Javascript or C# for development, I decided to use Unity with C# for the Project. I worked in a team of 6 for this Project (More details about team members available on Github repo page). We decided to build an educational and fun to play, catch and fall video game and named it Colorspot. To make it more interesting and educational, we added different color pallettes for each level.  
My team and I decided to go with the Agile software development lifecycle and used Trello board to track use cases.

After finishing the design stage, we decided to breakdown the development into two parts: Menu User Interface (UI) and Gameplay to make it easier to manage progress and to divide work amongst ourselves.
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

The Gameplay is divided into three scenes for three levels available:
<ul>
    <li>Level 1</li>
    <li>Level 2</li>
    <li>Level 3</li>
</ul>
Each level can be played for 60 seconds with a multiplier which increments after a caught node and resets on a missed note. The top panel on the level scene shows timer on the left, color pallete in the center and pause option on the right. The bottom panel shows the sprite selected, colors available in the pallette and the multiplier in order respectively. The level color pallette is applied to the falling notes. Level 1 contains primary colors, level 2 contains secondary colors and level 3 contains tertiary colors.This feature was added to make the game educational. The pause button pauses the timer and shows options to <i>Resume game</i>, <i>Quit game</i>, <i>Home</i> and <i>Restart</i>. The game ends when the timer reaches 0 after which a Final score is shown to the user. If the score exceeds 150, the players gets bonus points added to encourage user to catch as many notes as possible. If the user scores a high score, it is added to local disk for persistance. The user also earns coins equal to half the score which can be used to purchase items in shop.

---

The game demo is shown in the video below:
<!--Add youtube video snippet here-->

---

<a href="https://github.com/kasim95/Unity_Colorspot"> 
    Github Repo
</a>

---