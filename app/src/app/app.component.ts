import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'reconnaissance-facial';
  block = false;

  onAfficherblock() {
    this.block = !this.block;
  }
}
