---
title: "Colorspot"
date: 2020-02-13
tags: [unity, games, game development, C#]
header: 
    overlay_color: "#ffffff"
    overlay_image: "/assets/images/bgs/feria.png"
    overlay_filter: .4
    caption: "Photo Credit: [**Ines**](http://simpledesktops.com/browse/desktops/2014/jul/13/feria/)"
excerpt: "A catch game developed with Unity Engine, C# & Adobe Illustrator"
classes: wide
gallery:
  - url: "/assets/images/posts/colorspot-1.jpg"
    image_path: "/assets/images/posts/colorspot-1.jpg" 
    alt: "Menu"
    title: "Menu"
  - url: "/assets/images/posts/colorspot-2.jpg"
    image_path: "/assets/images/posts/colorspot-2.jpg"
    alt: "Shop"
    title: "Shop"
  - url: "/assets/images/posts/colorspot-3.jpg"
    image_path: "/assets/images/posts/colorspot-3.jpg"
    alt: "Gameplay"
    title: "Gameplay"
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

<b>Code:</b> <a href="https://github.com/kasim95/Unity_Colorspot" target="_blank" style="color: black;">Github repo <i class="fab fa-github fa-lg"></i></a>

---

This project involved developing a catch game using Unity Engine and C# in a team of six.
The visual assets for this project were created using Adobe Illustrator.
The team members can be found on the Readme page of the Github repository.
The development stage of the Project was broken down into two sections: 
Menu User Interface (UI) and Gameplay.

<b> User Interface</b>

The User Interface was split into five different scenes: 
<ul>
    <li>
        <b>Load User</b>: The Load User scene is used to select between 
        users. If a user does not exist denoted by < None >, a new user can be added onclick
        of the < None > button. To limit the storage used on disk, the number of users 
        is limited to 4. To streamline the process of 
        serializing user profile to/from disk, a <i>User</i> class is used. When the user 
        selects one of the profiles, the <code>makecurrentuser</code> function in User 
        class deserializes the user profile as a <i>User</i> object.
    </li>
    <li>
        <b>Level Select</b>: The Level Select scene contains options for <i>Quit</i>, 
        <i>Previous level</i>, <i>Next level</i>, <i>Home</i> and <i>Level Selectors</i>. 
        When a user clicks on a Level Selector, it changes the current scene to the game 
        level selected out of Level 1, 2 or 3 scenes. 
    </li>
    <li>
        <b>Menu</b>: The Menu scene contains options for <i>Change User</i>, 
        <i>Start game</i>, <i>Store</i>, <i>Settings</i> and <i>Exit</i>.
        The buttons are mapped to their respective scenes explained by their literal names 
        except the Exit button which is used to close the game.
    </li>
    <li>
        <b>Settings</b>: The Settings scene contains inputs for volume control which can 
        be changed using the slider available in the scene.
    </li>
    <li>
        <b>Shop</b>: The Shop Scene contains customization options for the game background,
        sprite (the catcher), and notes (the falling objects). The purchases made by user
        for customizations are serialized to user profile on disk for persistance.
    </li>
</ul>

<b>Gameplay</b>

The Gameplay section is divided into three scenes for three levels available:
<ul>
    <li>Level 1</li>
    <li>Level 2</li>
    <li>Level 3</li>
</ul>
Each level can be played for 60 seconds with a multiplier which increments after a note 
is caught and resets on a missed note. The top panel on the level scene shows timer on 
the left, color pallete in the center and pause option on the right. The bottom panel 
shows the sprite selected, colors available in the pallette and the multiplier in order 
respectively. The level color pallette is applied to the falling notes. Level 1 contains 
primary colors, level 2 contains secondary colors and level 3 contains tertiary colors. 
This feature was added to make the game educational. The pause button pauses the timer 
and shows options to <i>Resume game</i>, <i>Quit game</i>, <i>Home</i> and <i>Restart</i>. 
The game ends when the timer reaches 0 after which a final score is shown to the user. 
If the score exceeds 150, the players gets bonus points added to encourage user to catch 
as many notes as possible. If the user achieves a high score, it is persisted on disk. 
The user also earns coins after each game equal to 50% of the total score
These coins can be used to purchase items in shop.

Below are some screenshots for the game:

{% include gallery %}

<b>Tools Used</b>
<ul>
    <li>Unity</li>
    <li>C#</li>
    <li>Adobe Illustrator</li>
</ul>

---

<b>Demo</b>

<iframe width="560" height="315" src="https://www.youtube.com/embed/CNAdpm5gDwc" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
