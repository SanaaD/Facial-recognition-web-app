import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ApiResult } from './api-result';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  constructor(private httpClient: HttpClient) { }

  apiUrl = "api/";

  identify(data: Object) {
    return this.httpClient.post<ApiResult>(this.apiUrl + "identify", data);
  }

  add(data: Object) {
    return this.httpClient.post(this.apiUrl + "add", data);
  }
}
